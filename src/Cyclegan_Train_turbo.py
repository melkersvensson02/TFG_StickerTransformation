import os
import gc
import copy
import lpips
import torch
import torch.nn.functional as F
import wandb
from pathlib import Path
from glob import glob
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from diffusers.optimization import get_scheduler
from peft.utils import get_peft_model_state_dict
from cleanfid.fid import get_folder_features, build_feature_extractor, frechet_distance
import vision_aided_loss
import json
from model import make_1step_sched
from cyclegan_turbo import CycleGAN_Turbo, VAE_encode, VAE_decode, initialize_unet, initialize_vae
from my_utils.training_utils import UnpairedDataset, build_transform, parse_args_unpaired_training
from my_utils.dino_struct import DinoStructureLoss


def gradient_difference_loss(img1, img2, dtype=torch.float32):
    """
    Gradient Difference Loss using Sobel operators.
    Measures difference in gradients (edges) between two images.
    
    FIXED: Added dtype parameter to handle mixed precision training properly.
    Sobel kernels are created directly in the target dtype to avoid conversion issues.
    """
    # FIXED: Create Sobel kernels directly in target dtype and device
    gx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=dtype, device=img1.device)
    gy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=dtype, device=img1.device)
    gx = gx.view(1, 1, 3, 3)
    gy = gy.view(1, 1, 3, 3)
    
    # Convert to grayscale if needed (take mean across channels)
    if img1.shape[1] == 3:
        img1_gray = img1.mean(dim=1, keepdim=True)
        img2_gray = img2.mean(dim=1, keepdim=True)
    else:
        img1_gray = img1
        img2_gray = img2
    
    # Compute gradients
    grad1_x = F.conv2d(img1_gray, gx, padding=1)
    grad1_y = F.conv2d(img1_gray, gy, padding=1)
    grad2_x = F.conv2d(img2_gray, gx, padding=1)
    grad2_y = F.conv2d(img2_gray, gy, padding=1)
    
    # L1 loss on gradients
    loss = F.l1_loss(grad1_x, grad2_x) + F.l1_loss(grad1_y, grad2_y)
    return loss

def extract_mask_from_image(img, threshold=1.0):

    # Convert [-1, 1] to [0, 1]
    img_normalized = (img + 1) / 2
    
    # Check if all channels are close to 1.0 (pure white)
    # Use threshold slightly below 1.0 to account for minor variations
    mask = (img_normalized.mean(dim=1, keepdim=True) < 0.99).float()
    
    return mask
def apply_mask_to_image(img, mask, background_value=1.0):

    masked_img = img * mask + (1 - mask) * background_value
    return masked_img

class ContextualLoss:
    """
    Contextual Loss using VGG relu3_3 features.
    Measures patch-wise cosine similarity with max-pooling over patches.
    
    FIXED: VGG16 kept in float32 for numerical stability.
    Input images are converted to float32, processed, then results converted back.
    """
    def __init__(self, device="cuda", dtype=torch.float32):
        self.device = device
        self.dtype = dtype  # FIXED: Store target dtype for mixed precision
        # Load pretrained VGG16
        import torchvision.models as models
        # FIXED: Explicitly keep VGG in float32 for stability
        vgg16 = models.vgg16(pretrained=True).to(device).to(torch.float32)
        vgg16.eval()
        for param in vgg16.parameters():
            param.requires_grad = False
        
        # Extract up to relu3_3 (layer 16 in VGG16 features)
        self.features_extractor = torch.nn.Sequential(*list(vgg16.features.children())[:16]).to(device).to(torch.float32)
        self.features_extractor.eval()
    
    def forward(self, img1, img2):
        """
        Compute contextual loss between img1 and img2.
        img1, img2: tensors of shape (B, C, H, W) with values in [-1, 1]
        
        FIXED: Convert inputs to float32 for VGG processing to handle mixed precision.
        """
        # FIXED: Convert to float32 for VGG processing (numerical stability)
        img1 = img1.float()
        img2 = img2.float()
        
        # Normalize to [0, 1] and then to ImageNet stats
        img1_norm = (img1 + 1) / 2  # [-1, 1] -> [0, 1]
        img2_norm = (img2 + 1) / 2
        
        # FIXED: Create tensors in float32 explicitly
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        img1_norm = (img1_norm - mean) / std
        img2_norm = (img2_norm - mean) / std
        
        with torch.no_grad():
            feat1 = self.features_extractor(img1_norm)  # (B, 256, H/8, W/8)
            feat2 = self.features_extractor(img2_norm)
        
        # Reshape to (B, C, -1) for patch-wise operations
        B, C, H, W = feat1.shape
        feat1_patches = feat1.view(B, C, -1)  # (B, 256, N_patches)
        feat2_patches = feat2.view(B, C, -1)
        
        # Normalize for cosine similarity
        feat1_patches = F.normalize(feat1_patches, dim=1)  # (B, 256, N_patches)
        feat2_patches = F.normalize(feat2_patches, dim=1)
        
        # Compute cosine similarity matrix (B, N_patches, N_patches)
        # sim[b, i, j] = cosine similarity between patch i in feat1 and patch j in feat2
        sim = torch.bmm(feat1_patches.permute(0, 2, 1), feat2_patches)  # (B, N_patches, N_patches)
        
        # Max-pooling: for each patch in feat1, find best match in feat2
        max_sim_per_patch = sim.max(dim=2)[0]  # (B, N_patches)
        
        # Contextual loss: negative log of average max similarity
        cx_loss = -torch.log(max_sim_per_patch.mean() + 1e-5)
        
        return cx_loss


def main(args):
    # create one text file and one json file at the output dir
    os.makedirs(args.output_dir, exist_ok=True)
    JSON_FILE = os.path.join(args.output_dir, "metrics.jsonl")
    metrics_file = JSON_FILE
    TEXT_FILE = os.path.join(args.output_dir, "runing_information.txt") 
    accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,  
    log_with=args.report_to
    )
    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # Tokenizer and text encoder initialization (used for text conditioning)
    tokenizer = AutoTokenizer.from_pretrained("/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo", subfolder="tokenizer", revision=args.revision, use_fast=False,)
    noise_scheduler_1step = make_1step_sched()
    text_encoder = CLIPTextModel.from_pretrained("/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo", subfolder="text_encoder").cuda()
    # Initalize the UNet (here we add the LoRA layers as well) need to change path name inside here to local 
    unet, l_modules_unet_encoder, l_modules_unet_decoder, l_modules_unet_others = initialize_unet(args.lora_rank_unet, return_lora_module_names=True)
    # Initalize the VAE (here we add the LoRA layers as well) need to change path name inside here to local
    vae_a2b, vae_lora_target_modules = initialize_vae(args.lora_rank_vae, return_lora_module_names=True, ignore_skip=args.vae_ignore_skip, skip_weight=args.vae_skip_weight)
    write_text(TEXT_FILE, f"Pased UNET and VAE initialization \n")
    
    # Determine weight dtype for mixed precision training
    # NOTE: With Accelerator mixed_precision enabled, models stay in float32 (master weights)
    # but Accelerator's autocast automatically uses fp16/bf16 for operations during forward pass.
    # This weight_dtype variable is used for:
    #   1. Input data tensors (img_a, img_b) - converts to fp16/bf16 for performance
    #   2. GDL Sobel kernel creation (matches input dtype)
    #   3. Logging/reference (what dtype operations will use)
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    
    # Move models to CUDA (but keep them in float32)
    # Accelerator will manage dtype conversions via autocast during forward pass
    text_encoder.cuda()
    unet.cuda()
    vae_a2b.cuda()
    text_encoder.requires_grad_(False)
    
    # Initialize discriminators
    # NOTE: Discriminators stay in float32, Accelerator's autocast manages dtype conversions
    # CLIP backbone (frozen) operations stay float32 for stability
    # Discriminator head (trainable) operations use fp16/bf16 via autocast
    if args.gan_disc_type == "vagan_clip":
        net_disc_a = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
        net_disc_a.cv_ensemble.requires_grad_(False)  # Freeze CLIP feature extractor
        net_disc_b = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
        net_disc_b.cv_ensemble.requires_grad_(False)  # Freeze CLIP feature extractor

    # Define loss functions
    crit_cycle, crit_idt = torch.nn.L1Loss(), torch.nn.L1Loss()
    if args.enable_xformers_memory_efficient_attention:
        unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Make sure that the first conv layer of UNet is trainable IMP
    unet.conv_in.requires_grad_(True)
    # Copy VAE for the other direction
    vae_b2a = copy.deepcopy(vae_a2b)

    # IMP, this returns all the LoRA plus first layer of UNet 
    params_gen = CycleGAN_Turbo.get_traininable_params(unet, vae_a2b, vae_b2a)
    # vae_enc and vae_dec are just wrappers to call the VAE encode/decode with the correct VAE
    vae_enc = VAE_encode(vae_a2b, vae_b2a=vae_b2a)
    vae_dec = VAE_decode(vae_a2b, vae_b2a=vae_b2a)
    # Initialize ADAM for the trainable parameters of unet and vae
    optimizer_gen = torch.optim.AdamW(params_gen, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,)
    # Initialize ADAM for the discriminators
    params_disc = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
    optimizer_disc = torch.optim.AdamW(params_disc, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,)
    
    write_text(TEXT_FILE, f"\nNumber of parameters to optimize: Generator: {sum(p.numel() for p in params_gen if p.requires_grad)} \n")
    write_text(TEXT_FILE, f"Number of parameters to optimize: Discriminator: {sum(p.numel() for p in params_disc if p.requires_grad)} \n")
    write_text(TEXT_FILE, f"VAE LoRA target modules: {vae_lora_target_modules} \n")

    # Initialize dataloader 
    # The dataset folder structure is assumed to be: train_A, train_B, and fixed prompts for a/b  
    # dataset is built from the folder specified by args.dataset_folder and uses whatever image-preprocessing pipeline is defined in args.train_img_prep
    dataset_train = UnpairedDataset(dataset_folder=args.dataset_folder, image_prep=args.train_img_prep, split="train", tokenizer=tokenizer)
    # loads batches of training data of size args.train_batch_size, training will lop over these batches
    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    T_val = build_transform(args.val_img_prep)
    # get the promopts (in our case its just one line per domain)
    fixed_caption_src = dataset_train.fixed_caption_src
    fixed_caption_tgt = dataset_train.fixed_caption_tgt
    write_text(TEXT_FILE, f"\nDataLoaders created: train size={len(dataset_train)}\n")
    l_images_src_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_src_test.extend(glob(os.path.join(args.dataset_folder, "test_A", ext)))
    l_images_tgt_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_tgt_test.extend(glob(os.path.join(args.dataset_folder, "test_B", ext)))
    l_images_src_test, l_images_tgt_test = sorted(l_images_src_test), sorted(l_images_tgt_test)

    # make the reference FID statistics
    if accelerator.is_main_process:
        # Tool to mesure how well style is being transferred
        feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)
        """
        FID reference statistics for A -> B translation
        """
        output_dir_ref = os.path.join(args.output_dir, "fid_reference_a2b")
        os.makedirs(output_dir_ref, exist_ok=True)
        # transform all images according to the validation transform and save them
        for _path in tqdm(l_images_tgt_test):
            outf = os.path.join(output_dir_ref, os.path.basename(_path)).replace(".jpg", ".png")
            if not os.path.exists(outf):
                _img = T_val(Image.open(_path).convert("RGB"))
                _img.save(outf)
        # compute the features for the reference images (this is used later to compute FID)
        ref_features = get_folder_features(output_dir_ref, model=feat_model, num_workers=0, num=None,
                        shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                        mode="clean", custom_fn_resize=None, description="", verbose=True,
                        custom_image_tranform=None)
        a2b_ref_mu, a2b_ref_sigma = np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False)
        """
        FID reference statistics for B -> A translation
        """
        # transform all images according to the validation transform and save them
        output_dir_ref = os.path.join(args.output_dir, "fid_reference_b2a")
        os.makedirs(output_dir_ref, exist_ok=True)
        for _path in tqdm(l_images_src_test):
            outf = os.path.join(output_dir_ref, os.path.basename(_path)).replace(".jpg", ".png")
            if not os.path.exists(outf):
                _img = T_val(Image.open(_path).convert("RGB"))
                _img.save(outf)
        # compute the features for the reference images
        ref_features = get_folder_features(output_dir_ref, model=feat_model, num_workers=0, num=None,
                        shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                        mode="clean", custom_fn_resize=None, description="", verbose=True,
                        custom_image_tranform=None)
        b2a_ref_mu, b2a_ref_sigma = np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False)
    

    # Create LR schedulers for generator and discriminator optimizers
    lr_scheduler_gen = get_scheduler(args.lr_scheduler, optimizer=optimizer_gen,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power)
    
    # Initialize LPIPS model for perceptual loss
    # NOTE: LPIPS will be moved to weight_dtype after accelerator.prepare()
    # But inputs are still converted to .float() at call sites for numerical stability
    # (following pix2pix approach at line 388-389)
    net_lpips = lpips.LPIPS(net='vgg')
    net_lpips.cuda()
    net_lpips.requires_grad_(False)
    write_text(TEXT_FILE, f"LPIPS model initialized \n")
    
    # Initialize Contextual Loss if needed
    net_cx = None
    if args.lambda_cx > 0:
        # FIXED: Pass dtype parameter for mixed precision handling
        net_cx = ContextualLoss(device="cuda", dtype=weight_dtype)
        write_text(TEXT_FILE, f"Contextual Loss model initialized \n")
    
    # Precompute the fixed text embeddings for the fixed captions
    fixed_a2b_tokens = tokenizer(fixed_caption_tgt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
    fixed_a2b_emb_base = text_encoder(fixed_a2b_tokens.cuda().unsqueeze(0))[0].detach()
    fixed_b2a_tokens = tokenizer(fixed_caption_src, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
    fixed_b2a_emb_base = text_encoder(fixed_b2a_tokens.cuda().unsqueeze(0))[0].detach()
    del text_encoder, tokenizer  # free up some memory

    # Prepare everything with accelerator
    # Prepare trainable models and optimizers with accelerator (enables mixed precision, distributed training)
    unet, vae_enc, vae_dec, net_disc_a, net_disc_b = accelerator.prepare(unet, vae_enc, vae_dec, net_disc_a, net_disc_b)
    net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc = accelerator.prepare(
        net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc
    )
    
    # CRITICAL: Move models to weight_dtype AFTER accelerator.prepare()
    # This matches pix2pix approach (Train_pix2pix_turbo.py lines 286-289)
    # We MUST convert models to target dtype for xformers compatibility with Q/K/V
    unet.to(dtype=weight_dtype)
    vae_enc.to(dtype=weight_dtype)
    vae_dec.to(dtype=weight_dtype)
    net_disc_a.to(dtype=weight_dtype)
    net_disc_b.to(dtype=weight_dtype)
    
    # LPIPS: Keep in float32 to match .float() input conversions
    # VGG-based models are more stable in float32 anyway
    net_lpips.to(dtype=torch.float32)
    
    write_text(TEXT_FILE, f"\nTrainable models converted to dtype={weight_dtype}\n")
    write_text(TEXT_FILE, f"LPIPS kept in float32 for numerical stability\n")
    
    # Starts logging metrics/training progress to Weights & Biases, TensorBoard, or similar
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)), 
                             init_kwargs={"wandb": {"mode": "offline"}})

    first_epoch = 0
    global_step = 0
    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,)
    
    # turn off eff. attn for the disc (likely because discriminator does not benefit from it)
    # this is a more of a stability safeguard
    # notice how in contrast to the paired case, where we did cv_ensemble.requires_grad_(False)
    # this is not done becuase the discriminator needs to learn to distinguish real vs fake
    for name, module in net_disc_a.named_modules():
        if "attn" in name:
            module.fused_attn = False
    for name, module in net_disc_b.named_modules():
        if "attn" in name:
            module.fused_attn = False

    for epoch in range(first_epoch, args.max_train_epochs):
        for step, batch in enumerate(train_dataloader):
            l_acc = [unet, net_disc_a, net_disc_b, vae_enc, vae_dec]
            with accelerator.accumulate(*l_acc):
                # Get the input images
                img_a = batch["pixel_values_src"].to(dtype=weight_dtype)
                img_b = batch["pixel_values_tgt"].to(dtype=weight_dtype)

                bsz = img_a.shape[0]
                # Precompute the text embeddings for the batch captions
                fixed_a2b_emb = fixed_a2b_emb_base.repeat(bsz, 1, 1).to(dtype=weight_dtype)
                fixed_b2a_emb = fixed_b2a_emb_base.repeat(bsz, 1, 1).to(dtype=weight_dtype)
                timesteps = torch.tensor([noise_scheduler_1step.config.num_train_timesteps - 1] * bsz, device=img_a.device).long()

                """
                Cycle Objective
                """
                # A -> fake B -> rec A 
                # NOTICE: UNet is the same for both directions, this is a concious design choice 
                # The VAEs learn to output latents that have structural and style information 
                # The UNet learns to interpret those latents and apply the right transformation 
                cyc_fake_b = CycleGAN_Turbo.forward_with_networks(img_a, "a2b", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_a2b_emb)
                cyc_rec_a = CycleGAN_Turbo.forward_with_networks(cyc_fake_b, "b2a", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_b2a_emb)
                # Extract mask from img_a (the reference in domain A)
                mask_a = extract_mask_from_image(img_a)
                # Apply mask to reconstructed image
                cyc_rec_a_masked = apply_mask_to_image(cyc_rec_a, mask_a, background_value=1.0)
                # Compute Lrec on masked reconstruction
                loss_cycle_a = crit_cycle(cyc_rec_a_masked, img_a) * args.lambda_cycle
                # Convert to float for LPIPS (following pix2pix approach for numerical stability)
                loss_cycle_a += net_lpips(cyc_rec_a_masked.float(), img_a.float()).mean() * args.lambda_cycle_lpips
                write_text(TEXT_FILE, f"Loss cycle A: {loss_cycle_a}\n")
                if args.lambda_gdl > 0:
                    loss_cycle_a += gradient_difference_loss(cyc_rec_a_masked, img_a, dtype=weight_dtype) * args.lambda_gdl
                if net_cx is not None and args.lambda_cx > 0:
                    loss_cycle_a += net_cx.forward(cyc_rec_a_masked, img_a) * args.lambda_cx
                # B -> fake A -> rec B
                cyc_fake_a = CycleGAN_Turbo.forward_with_networks(img_b, "b2a", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_b2a_emb)
                cyc_rec_b = CycleGAN_Turbo.forward_with_networks(cyc_fake_a, "a2b", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_a2b_emb)
                loss_cycle_b = crit_cycle(cyc_rec_b, img_b) * args.lambda_cycle
                # Convert to float for LPIPS (following pix2pix approach for numerical stability)
                loss_cycle_b += net_lpips(cyc_rec_b.float(), img_b.float()).mean() * args.lambda_cycle_lpips
                if args.lambda_gdl > 0:
                    loss_cycle_b += gradient_difference_loss(cyc_rec_b, img_b, dtype=weight_dtype) * args.lambda_gdl
                if net_cx is not None and args.lambda_cx > 0:
                    loss_cycle_b += net_cx.forward(cyc_rec_b, img_b) * args.lambda_cx
                write_text(TEXT_FILE, f"Loss cycle B: {loss_cycle_b}\n")
                # The models are wrapped by accelerate for distributed/FP16 training.
                accelerator.backward(loss_cycle_a + loss_cycle_b, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                # Applies those gradients to the model weights — this is the actual update.
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                """
                Generator Objective (GAN) for task a->b and b->a (fake inputs)
                """
                fake_a = CycleGAN_Turbo.forward_with_networks(img_b, "b2a", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_b2a_emb)
                fake_b = CycleGAN_Turbo.forward_with_networks(img_a, "a2b", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_a2b_emb)
                # Discriminator can handle weight_dtype directly (following pix2pix approach)
                loss_gan_a = net_disc_a(fake_b, for_G=True).mean() * args.lambda_gan
                write_text(TEXT_FILE, f"Loss GAN A: {loss_gan_a}\n")
                loss_gan_b = net_disc_b(fake_a, for_G=True).mean() * args.lambda_gan
                write_text(TEXT_FILE, f"Loss GAN B: {loss_gan_b}\n")
                accelerator.backward(loss_gan_a + loss_gan_b, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()
                optimizer_disc.zero_grad()

                """
                Identity Objective
                """
                idt_a = CycleGAN_Turbo.forward_with_networks(img_b, "a2b", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_a2b_emb)
                loss_idt_a = crit_idt(idt_a, img_b) * args.lambda_idt
                # Convert to float for LPIPS (following pix2pix approach for numerical stability)
                loss_idt_a += net_lpips(idt_a.float(), img_b.float()).mean() * args.lambda_idt_lpips
                if args.lambda_gdl > 0:
                    loss_idt_a += gradient_difference_loss(idt_a, img_b, dtype=weight_dtype) * args.lambda_gdl
                if net_cx is not None and args.lambda_cx > 0:
                    loss_idt_a += net_cx.forward(idt_a, img_b) * args.lambda_cx
                idt_b = CycleGAN_Turbo.forward_with_networks(img_a, "b2a", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_b2a_emb)
                loss_idt_b = crit_idt(idt_b, img_a) * args.lambda_idt
                # Convert to float for LPIPS (following pix2pix approach for numerical stability)
                loss_idt_b += net_lpips(idt_b.float(), img_a.float()).mean() * args.lambda_idt_lpips
                write_text(TEXT_FILE, f"Loss IDT A: {loss_idt_a}\n")
                if args.lambda_gdl > 0:
                    loss_idt_b += gradient_difference_loss(idt_b, img_a, dtype=weight_dtype) * args.lambda_gdl
                if net_cx is not None and args.lambda_cx > 0:
                    loss_idt_b += net_cx.forward(idt_b, img_a) * args.lambda_cx
                loss_g_idt = loss_idt_a + loss_idt_b
                accelerator.backward(loss_g_idt, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                """
                Discriminator for task a->b and b->a (fake inputs)
                """
                # Discriminator can handle weight_dtype directly (following pix2pix approach)
                loss_D_A_fake = net_disc_a(fake_b.detach(), for_real=False).mean() * args.lambda_gan
                loss_D_B_fake = net_disc_b(fake_a.detach(), for_real=False).mean() * args.lambda_gan
                write_text(TEXT_FILE, f"Loss D A fake: {loss_D_A_fake}\n")
                write_text(TEXT_FILE, f"Loss D B fake: {loss_D_B_fake}\n")
                loss_D_fake = (loss_D_A_fake + loss_D_B_fake) * 0.5
                accelerator.backward(loss_D_fake, retain_graph=False)
                if accelerator.sync_gradients:
                    params_to_clip = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

                """
                Discriminator for task a->b and b->a (real inputs)
                """
                # Discriminator can handle weight_dtype directly (following pix2pix approach)
                loss_D_A_real = net_disc_a(img_b, for_real=True).mean() * args.lambda_gan
                loss_D_B_real = net_disc_b(img_a, for_real=True).mean() * args.lambda_gan
                write_text(TEXT_FILE, f"Loss D A real: {loss_D_A_real}\n")
                write_text(TEXT_FILE, f"Loss D B real: {loss_D_B_real}\n")
                loss_D_real = (loss_D_A_real + loss_D_B_real) * 0.5
                accelerator.backward(loss_D_real, retain_graph=False)
                if accelerator.sync_gradients:
                    params_to_clip = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

            logs = {}
            # FIXED: Check for NaN in tensors BEFORE converting to .item()
            if torch.isnan(loss_cycle_a).any() and globle_step % 10 == 9:  # Only check after 9 steps 
                write_text(TEXT_FILE, f"\n{'='*80}\n")
                write_text(TEXT_FILE, f"ERROR: NaN detected in loss_cycle_a at step {global_step}\n")
                write_text(TEXT_FILE, f"  img_a stats: min={img_a.min().item()}, max={img_a.max().item()}, mean={img_a.mean().item()}\n")
                write_text(TEXT_FILE, f"  mask_a stats: min={mask_a.min().item()}, max={mask_a.max().item()}, mean={mask_a.mean().item()}\n")
                write_text(TEXT_FILE, f"  cyc_rec_a_masked stats: min={cyc_rec_a_masked.min().item()}, max={cyc_rec_a_masked.max().item()}, mean={cyc_rec_a_masked.mean().item()}\n")
                write_text(TEXT_FILE, f"{'='*80}\n")
                raise RuntimeError(f"EXECUTION STOPPED: NaN in loss_cycle_a at step {global_step}")
            
            logs["cycle_a"] = loss_cycle_a.detach().item()
            logs["cycle_b"] = loss_cycle_b.detach().item()
            logs["gan_a"] = loss_gan_a.detach().item()
            logs["gan_b"] = loss_gan_b.detach().item()
            logs["disc_a"] = loss_D_A_fake.detach().item() + loss_D_A_real.detach().item()
            logs["disc_b"] = loss_D_B_fake.detach().item() + loss_D_B_real.detach().item()
            logs["idt_a"] = loss_idt_a.detach().item()
            logs["idt_b"] = loss_idt_b.detach().item()
            
            # FIXED: Comprehensive NaN detection with execution stopping
            nan_detected = False
            nan_keys = []
            for key, value in logs.items():
                if np.isnan(value):
                    nan_detected = True
                    nan_keys.append(key)
            
            if nan_detected:
                write_text(TEXT_FILE, f"\n{'='*80}\n")
                write_text(TEXT_FILE, f"ERROR: NaN detected at step {global_step}\n")
                write_text(TEXT_FILE, f"NaN in losses: {nan_keys}\n")
                write_text(TEXT_FILE, f"All loss values: {logs}\n")
                write_text(TEXT_FILE, f"\nDiagnostics:\n")
                write_text(TEXT_FILE, f"  img_a stats: min={img_a.min().item()}, max={img_a.max().item()}, mean={img_a.mean().item()}\n")
                write_text(TEXT_FILE, f"  img_b stats: min={img_b.min().item()}, max={img_b.max().item()}, mean={img_b.mean().item()}\n")
                write_text(TEXT_FILE, f"  mask_a stats: min={mask_a.min().item()}, max={mask_a.max().item()}, mean={mask_a.mean().item()}\n")
                write_text(TEXT_FILE, f"  cyc_fake_b stats: min={cyc_fake_b.min().item()}, max={cyc_fake_b.max().item()}, mean={cyc_fake_b.mean().item()}\n")
                write_text(TEXT_FILE, f"  cyc_rec_a stats: min={cyc_rec_a.min().item()}, max={cyc_rec_a.max().item()}, mean={cyc_rec_a.mean().item()}\n")
                write_text(TEXT_FILE, f"  cyc_rec_a_masked stats: min={cyc_rec_a_masked.min().item()}, max={cyc_rec_a_masked.max().item()}, mean={cyc_rec_a_masked.mean().item()}\n")
                write_text(TEXT_FILE, f"  weight_dtype: {weight_dtype}\n")
                write_text(TEXT_FILE, f"{'='*80}\n")
                raise RuntimeError(f"EXECUTION STOPPED: NaN detected in losses {nan_keys} at step {global_step}. Check diagnostics in {TEXT_FILE}")
            
            # REMOVED: Unnecessary del statements - Python garbage collector handles this
            write_text(TEXT_FILE, f"Step {logs}: wrote metrics")

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    eval_unet = accelerator.unwrap_model(unet)
                    # Get trainable parameters of from encoder in both directions
                    eval_vae_enc = accelerator.unwrap_model(vae_enc)
                    eval_vae_dec = accelerator.unwrap_model(vae_dec)
                    if global_step % args.viz_freq == 1:
                        for tracker in accelerator.trackers:
                            if tracker.name == "wandb":
                                viz_img_a = batch["pixel_values_src"].to(dtype=weight_dtype)
                                viz_img_b = batch["pixel_values_tgt"].to(dtype=weight_dtype)
                                # FIXED: Convert to float32, normalize to [0,1], and clamp for proper visualization
                                log_dict = {
                                    "train/real_a": [wandb.Image((viz_img_a[idx].float() * 0.5 + 0.5).clamp(0, 1).detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)],
                                    "train/real_b": [wandb.Image((viz_img_b[idx].float() * 0.5 + 0.5).clamp(0, 1).detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)],
                                }
                                log_dict["train/rec_a"] = [wandb.Image((cyc_rec_a[idx].float() * 0.5 + 0.5).clamp(0, 1).detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)]
                                log_dict["train/rec_b"] = [wandb.Image((cyc_rec_b[idx].float() * 0.5 + 0.5).clamp(0, 1).detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)]
                                log_dict["train/fake_b"] = [wandb.Image((fake_b[idx].float() * 0.5 + 0.5).clamp(0, 1).detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)]
                                log_dict["train/fake_a"] = [wandb.Image((fake_a[idx].float() * 0.5 + 0.5).clamp(0, 1).detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)]
                                tracker.log(log_dict)
                                gc.collect()
                                torch.cuda.empty_cache()
                    # Save LoRA and VAE weights (when training is finished this will be the final model)
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        sd = {}
                        sd["l_target_modules_encoder"] = l_modules_unet_encoder
                        sd["l_target_modules_decoder"] = l_modules_unet_decoder
                        sd["l_modules_others"] = l_modules_unet_others
                        sd["rank_unet"] = args.lora_rank_unet
                        sd["sd_encoder"] = get_peft_model_state_dict(eval_unet, adapter_name="default_encoder")
                        sd["sd_decoder"] = get_peft_model_state_dict(eval_unet, adapter_name="default_decoder")
                        sd["sd_other"] = get_peft_model_state_dict(eval_unet, adapter_name="default_others")
                        sd["rank_vae"] = args.lora_rank_vae
                        sd["vae_lora_target_modules"] = vae_lora_target_modules
                        # This allows two identical copies of the VAE to coexist in one state_dict()
                        # Because wrapper VAE_encode stores both VAEs as self.vae and self.vae_b2a
                        # And this name distinction is used when generating the key names as the first part
                        # E.g "vae.encoder.conv1" vs "vae_b2a.encoder.conv1"
                        sd["sd_vae_enc"] = eval_vae_enc.state_dict()
                        # This saves the whole VAE weights (base + LoRA) for both VAEs
                        sd["sd_vae_dec"] = eval_vae_dec.state_dict()
                        torch.save(sd, outf)
                        gc.collect()
                        torch.cuda.empty_cache()

                    # compute val FID and DINO-Struct scores
                    if global_step % args.validation_steps == 1:
                        _timesteps = torch.tensor([noise_scheduler_1step.config.num_train_timesteps - 1] * 1, device="cuda").long()
                        net_dino = DinoStructureLoss()
                        """
                        Evaluate "A->B"
                        """
                        fid_output_dir = os.path.join(args.output_dir, f"fid-{global_step}/samples_a2b")
                        os.makedirs(fid_output_dir, exist_ok=True)
                        l_dino_scores_a2b = []
                        # get val input images from domain a
                        write_text(TEXT_FILE, f"Number of images to be processed {args.validation_num_images}")
                        for idx, input_img_path in enumerate(tqdm(l_images_src_test)):
                            if idx > args.validation_num_images and args.validation_num_images > 0:
                                break
                            orig_name = os.path.splitext(os.path.basename(input_img_path))[0]
                            outf = os.path.join(fid_output_dir, f"{orig_name}_fid_{global_step}.png")
                            with torch.no_grad():
                                input_img = T_val(Image.open(input_img_path).convert("RGB"))
                                #print(f"Input PIL size: {input_img.size}")
                                img_a = transforms.ToTensor()(input_img)
                                img_a = transforms.Normalize([0.5], [0.5])(img_a).unsqueeze(0).cuda().to(dtype=weight_dtype)
                                #print(f"img_a tensor shape: {img_a.shape}")
                                eval_fake_b = CycleGAN_Turbo.forward_with_networks(img_a, "a2b", eval_vae_enc, eval_unet,
                                    eval_vae_dec, noise_scheduler_1step, _timesteps, fixed_a2b_emb[0:1])
                                #print(f"eval_fake_b tensor shape: {eval_fake_b.shape}")
                                # FIXED: Convert to float32 and clamp before ToPILImage for mixed precision compatibility
                                eval_fake_b_pil = transforms.ToPILImage()((eval_fake_b[0].float() * 0.5 + 0.5).clamp(0, 1))
                                #print(f"Output PIL size: {eval_fake_b_pil.size}")
                                # Resize to match input ONLY for metrics
                                eval_fake_b_pil_for_metrics = eval_fake_b_pil.resize(input_img.size, Image.LANCZOS)
                                eval_fake_b_pil.save(outf)
                                # FIXED: Check if image is all black (catastrophic failure detection)
                                img_array = np.array(eval_fake_b_pil)
                                mean_intensity = img_array.mean()
                                if mean_intensity < 1.0:  # Essentially black image (0-255 scale)
                                    write_text(TEXT_FILE, f"\nERROR: FID image is all black at step {global_step}, file {outf}\n")
                                    write_text(TEXT_FILE, f"  Mean intensity: {mean_intensity}\n")
                                    write_text(TEXT_FILE, f"  eval_fake_b stats: min={eval_fake_b.min().item()}, max={eval_fake_b.max().item()}, mean={eval_fake_b.mean().item()}\n")
                                    raise RuntimeError(f"EXECUTION STOPPED: FID images are all black at step {global_step}. Check model outputs and dtype handling.")
                                a = net_dino.preprocess(input_img).unsqueeze(0).cuda()
                                b = net_dino.preprocess(eval_fake_b_pil_for_metrics).unsqueeze(0).cuda()
                                #print(f"DINO a shape: {a.shape}, DINO b shape: {b.shape}")
                                dino_ssim = net_dino.calculate_global_ssim_loss(a, b).item()
                                l_dino_scores_a2b.append(dino_ssim)
                        dino_score_a2b = np.mean(l_dino_scores_a2b)
                        gen_features = get_folder_features(fid_output_dir, model=feat_model, num_workers=0, num=None,
                            shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                            mode="clean", custom_fn_resize=None, description="", verbose=True,
                            custom_image_tranform=None)
                        ed_mu, ed_sigma = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
                        score_fid_a2b = frechet_distance(a2b_ref_mu, a2b_ref_sigma, ed_mu, ed_sigma)
                        #print(f"step={global_step}, fid(a2b)={score_fid_a2b:.2f}, dino(a2b)={dino_score_a2b:.3f}")

                        """
                        compute FID for "B->A"
                        """
                        fid_output_dir = os.path.join(args.output_dir, f"fid-{global_step}/samples_b2a")
                        os.makedirs(fid_output_dir, exist_ok=True)
                        l_dino_scores_b2a = []
                        # get val input images from domain b
                        for idx, input_img_path in enumerate(tqdm(l_images_tgt_test)):
                            if idx > args.validation_num_images and args.validation_num_images > 0:
                                break
                            orig_name = os.path.splitext(os.path.basename(input_img_path))[0]
                            outf = os.path.join(fid_output_dir, f"{orig_name}_fid_{global_step}.png")
                            with torch.no_grad():
                                input_img = T_val(Image.open(input_img_path).convert("RGB"))
                                img_b = transforms.ToTensor()(input_img)
                                img_b = transforms.Normalize([0.5], [0.5])(img_b).unsqueeze(0).cuda().to(dtype=weight_dtype)
                                eval_fake_a = CycleGAN_Turbo.forward_with_networks(img_b, "b2a", eval_vae_enc, eval_unet,
                                    eval_vae_dec, noise_scheduler_1step, _timesteps, fixed_b2a_emb[0:1])
                                # FIXED: Convert to float32 and clamp before ToPILImage for mixed precision compatibility
                                eval_fake_a_pil = transforms.ToPILImage()((eval_fake_a[0].float() * 0.5 + 0.5).clamp(0, 1))
                                eval_fake_a_pil.save(outf)
                                # FIXED: Check if image is all black (catastrophic failure detection)
                                img_array = np.array(eval_fake_a_pil)
                                mean_intensity = img_array.mean()
                                if mean_intensity < 1.0:  # Essentially black image (0-255 scale)
                                    write_text(TEXT_FILE, f"\nERROR: FID image is all black at step {global_step}, file {outf}\n")
                                    write_text(TEXT_FILE, f"  Mean intensity: {mean_intensity}\n")
                                    write_text(TEXT_FILE, f"  eval_fake_a stats: min={eval_fake_a.min().item()}, max={eval_fake_a.max().item()}, mean={eval_fake_a.mean().item()}\n")
                                    raise RuntimeError(f"EXECUTION STOPPED: FID images are all black at step {global_step}. Check model outputs and dtype handling.")
                                a = net_dino.preprocess(input_img).unsqueeze(0).cuda()
                                b = net_dino.preprocess(eval_fake_a_pil).unsqueeze(0).cuda()
                                dino_ssim = net_dino.calculate_global_ssim_loss(a, b).item()
                                l_dino_scores_b2a.append(dino_ssim)
                        dino_score_b2a = np.mean(l_dino_scores_b2a)
                        gen_features = get_folder_features(fid_output_dir, model=feat_model, num_workers=0, num=None,
                            shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                            mode="clean", custom_fn_resize=None, description="", verbose=True,
                            custom_image_tranform=None)
                        ed_mu, ed_sigma = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
                        score_fid_b2a = frechet_distance(b2a_ref_mu, b2a_ref_sigma, ed_mu, ed_sigma)
                        #print(f"step={global_step}, fid(b2a)={score_fid_b2a}, dino(b2a)={dino_score_b2a:.3f}")
                        logs["val/fid_a2b"], logs["val/fid_b2a"] = score_fid_a2b, score_fid_b2a
                        logs["val/dino_struct_a2b"], logs["val/dino_struct_b2a"] = dino_score_a2b, dino_score_b2a
                        del net_dino  # free up memory
                        # Save the parameters 
                        write_json_metrics(JSON_FILE, global_step, logs)
                        write_text(TEXT_FILE, f"Step {global_step}: wrote metrics")

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break


def write_json_metrics(file_path, step, logs):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    simple_logs = {}
    for k, v in logs.items():
        if isinstance(v, (int, float, str, bool)):
            simple_logs[k] = float(v) if isinstance(v, (int, float)) else v

    simple_logs["step"] = int(step)

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(simple_logs) + "\n")



            
def write_text(output_dir, text):
    """
    Append a line of text to output_dir/log.txt
    """
    with open(output_dir, "a", encoding="utf-8") as f:
        f.write(str(text))

if __name__ == "__main__":
    args = parse_args_unpaired_training()
    main(args)

