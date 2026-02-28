"""
Training script for fine-tuning a CycleGAN model using pix2pix setup on paired data.

This script is based on train_pix2pix_turbo.py with the following KEY DIFFERENCES:
1. Loads from CycleGAN checkpoint instead of initializing from scratch
2. Uses Pix2Pix_Turbo_from_CycleGAN model class (3 LoRA adapters for UNet)
3. Supports both per-image prompts and generic prompts via --data_set_type flag
4. Uses standard L2 loss (not weighted) to handle target domain displacement

All other aspects (discriminator, LPIPS, CLIP-sim, training loop) are identical to train_pix2pix_turbo.py
"""

import os
import gc
import lpips
import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import json
from pathlib import Path
import torchvision.utils as vutils
import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
import vision_aided_loss

from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats

# DIFFERENCE: Import our custom model class instead of Pix2Pix_Turbo
from pix2pix_turbo_from_cyclegan import Pix2Pix_Turbo_from_CycleGAN
from my_utils.training_utils import parse_args_paired_training, PairedDataset


def main(args):
    """
    Main training function.
    Structure identical to train_pix2pix_turbo.py with noted differences.
    """
    # Create output directories and log files
    os.makedirs(args.output_dir, exist_ok=True)
    JSON_FILE = os.path.join(args.output_dir, "metrics.jsonl")
    TEXT_FILE = os.path.join(args.output_dir, "running_information.txt")
    
    # Initialize accelerator for distributed training and mixed precision
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )
    
    write_text(TEXT_FILE, "="*80 + "\n")
    write_text(TEXT_FILE, "FINE-TUNING CYCLEGAN WITH PIX2PIX SETUP\n")
    write_text(TEXT_FILE, "="*80 + "\n\n")
    write_text(TEXT_FILE, f"CycleGAN checkpoint: {args.cyclegan_checkpoint}\n")
    write_text(TEXT_FILE, f"Dataset type: {args.data_set_type}\n")
    write_text(TEXT_FILE, f"Output directory: {args.output_dir}\n\n")
    
    write_text(
        TEXT_FILE,
        f"Accelerator: mixed_precision={accelerator.mixed_precision}, "
        f"num_processes={accelerator.num_processes}, "
        f"is_main={accelerator.is_local_main_process}\n\n"
    )
    
    # Set logging verbosity
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    # Set random seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
    
    # Create checkpoint and evaluation directories
    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
    
    # ============================================================================
    # DIFFERENCE: Initialize model from CycleGAN checkpoint
    # ============================================================================
    write_text(TEXT_FILE, "Initializing model from CycleGAN checkpoint...\n")
    
    net_pix2pix = Pix2Pix_Turbo_from_CycleGAN(
        cyclegan_checkpoint_path=args.cyclegan_checkpoint,
        lora_rank_unet=args.lora_rank_unet,
        lora_rank_vae=args.lora_rank_vae,
        ignore_skip=args.ignore_skip,
        skip_weight=args.skip_weight,
    )
    net_pix2pix.set_train()
    
    write_text(TEXT_FILE, "✓ Model initialized successfully!\n")
    write_text(
        TEXT_FILE,
        f"Total parameters: {sum(p.numel() for p in net_pix2pix.parameters()):,}\n"
        f"Trainable parameters: {sum(p.numel() for p in net_pix2pix.parameters() if p.requires_grad):,}\n\n"
    )
    
    # Enable xformers memory efficient attention (same as original)
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_pix2pix.unet.enable_xformers_memory_efficient_attention()
            write_text(TEXT_FILE, "✓ Enabled xformers memory efficient attention\n")
        else:
            raise ValueError(
                "xformers is not available, install with: pip install xformers"
            )
    
    # Enable gradient checkpointing to save memory (same as original)
    if args.gradient_checkpointing:
        net_pix2pix.unet.enable_gradient_checkpointing()
        write_text(TEXT_FILE, "✓ Enabled gradient checkpointing\n")
    
    # Allow TF32 for faster training on Ampere GPUs (same as original)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        write_text(TEXT_FILE, "✓ Enabled TF32\n")
    
    # ============================================================================
    # Initialize discriminator (same as original - starts from scratch)
    # ============================================================================
    write_text(TEXT_FILE, "\nInitializing discriminator (fresh weights)...\n")
    
    if args.gan_disc_type == "vagan_clip":
        net_disc = vision_aided_loss.Discriminator(
            cv_type="clip", loss_type=args.gan_loss_type, device="cuda"
        )
    else:
        raise NotImplementedError(
            f"Discriminator type {args.gan_disc_type} not implemented"
        )
    
    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)  # Freeze CLIP backbone
    net_disc.train()
    
    write_text(TEXT_FILE, "✓ Discriminator initialized\n")
    
    # ============================================================================
    # Initialize perceptual loss networks (same as original)
    # ============================================================================
    write_text(TEXT_FILE, "\nInitializing perceptual loss networks...\n")
    
    net_lpips = lpips.LPIPS(net="vgg").cuda()
    net_lpips.requires_grad_(False)
    write_text(TEXT_FILE, "✓ LPIPS (VGG) loaded\n")
    
    net_clip, _ = clip.load("ViT-B/32", device="cuda")
    net_clip.requires_grad_(False)
    net_clip.eval()
    write_text(TEXT_FILE, "✓ CLIP loaded\n\n")
    
    # ============================================================================
    # Collect trainable parameters (same as original)
    # ============================================================================
    write_text(TEXT_FILE, "Collecting trainable parameters...\n")
    
    layers_to_opt = []
    
    # UNet: LoRA layers + conv_in
    for n, p in net_pix2pix.unet.named_parameters():
        if "lora" in n:
            assert p.requires_grad
            layers_to_opt.append(p)
    layers_to_opt += list(net_pix2pix.unet.conv_in.parameters())
    
    # VAE: LoRA layers + skip convolutions
    for n, p in net_pix2pix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert p.requires_grad
            layers_to_opt.append(p)
    
    if not args.ignore_skip:
        layers_to_opt += list(net_pix2pix.vae.decoder.skip_conv_1.parameters())
        layers_to_opt += list(net_pix2pix.vae.decoder.skip_conv_2.parameters())
        layers_to_opt += list(net_pix2pix.vae.decoder.skip_conv_3.parameters())
        layers_to_opt += list(net_pix2pix.vae.decoder.skip_conv_4.parameters())
    
    write_text(
        TEXT_FILE,
        f"✓ Trainable parameters: {sum(p.numel() for p in layers_to_opt):,}\n\n"
    )
    
    # ============================================================================
    # Create optimizers and schedulers (same as original)
    # ============================================================================
    write_text(TEXT_FILE, "Creating optimizers...\n")
    
    # Generator (UNet + VAE) optimizer
    optimizer = torch.optim.AdamW(
        layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    
    # Discriminator optimizer
    optimizer_disc = torch.optim.AdamW(
        net_disc.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    lr_scheduler_disc = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    
    write_text(TEXT_FILE, "✓ Optimizers created\n\n")
    
    # ============================================================================
    # DIFFERENCE: Create datasets with data_set_type flag
    # ============================================================================
    write_text(TEXT_FILE, f"Creating datasets (type: {args.data_set_type})...\n")
    
    dataset_train = PairedDataset(
        dataset_folder=args.dataset_folder,
        image_prep=args.train_image_prep,
        split="train",
        tokenizer=net_pix2pix.tokenizer,
        data_set_type=args.data_set_type,  # "original" or "modified"
        use_canny_conditioning=args.use_canny_conditioning,
        canny_low_threshold=args.canny_low_threshold,
        canny_high_threshold=args.canny_high_threshold,
        binary_threshold=args.binary_threshold,
    )
    
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    
    dataset_val = PairedDataset(
        dataset_folder=args.dataset_folder,
        image_prep=args.test_image_prep,
        split="test",
        tokenizer=net_pix2pix.tokenizer,
        data_set_type=args.data_set_type,
        use_canny_conditioning=args.use_canny_conditioning,
        canny_low_threshold=args.canny_low_threshold,
        canny_high_threshold=args.canny_high_threshold,
        binary_threshold=args.binary_threshold,
    )
    
    dl_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=0
    )
    
    write_text(
        TEXT_FILE,
        f"✓ Train dataset: {len(dataset_train)} images\n"
        f"✓ Val dataset: {len(dataset_val)} images\n\n"
    )
    
    # ============================================================================
    # Save debug samples from dataloaders (same as original)
    # ============================================================================
    debug_dir = os.path.join(args.output_dir, "debug_dataloader")
    write_text(TEXT_FILE, f"Saving debug samples to: {debug_dir}\n")
    
    batch_train = next(iter(dl_train))
    save_debug_batch(batch_train, debug_dir, prefix="train_batch1", max_images=20)
    batch_train = next(iter(dl_train))
    save_debug_batch(batch_train, debug_dir, prefix="train_batch2", max_images=20)
    
    batch_val = next(iter(dl_val))
    save_debug_batch(batch_val, debug_dir, prefix="test_batch1", max_images=10)
    batch_val = next(iter(dl_val))
    save_debug_batch(batch_val, debug_dir, prefix="test_batch2", max_images=10)
    
    write_text(TEXT_FILE, "✓ Debug samples saved\n\n")
    
    # ============================================================================
    # Prepare models with accelerator (same as original)
    # ============================================================================
    write_text(TEXT_FILE, "Preparing models with accelerator...\n")
    
    (
        net_pix2pix,
        net_disc,
        optimizer,
        optimizer_disc,
        dl_train,
        lr_scheduler,
        lr_scheduler_disc,
    ) = accelerator.prepare(
        net_pix2pix,
        net_disc,
        optimizer,
        optimizer_disc,
        dl_train,
        lr_scheduler,
        lr_scheduler_disc,
    )
    
    net_clip, net_lpips = accelerator.prepare(net_clip, net_lpips)
    
    # CLIP normalization transform
    t_clip_renorm = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )
    
    # Set dtype based on mixed precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # Move networks to appropriate dtype
    net_pix2pix.to(dtype=weight_dtype)
    net_disc.to(dtype=weight_dtype)
    net_lpips.to(dtype=weight_dtype)
    net_clip.to(dtype=weight_dtype)
    
    write_text(TEXT_FILE, f"✓ Models prepared (dtype: {weight_dtype})\n\n")
    
    # ============================================================================
    # Initialize experiment tracking (same as original)
    # ============================================================================
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)
        write_text(TEXT_FILE, "✓ Tracker initialized\n")
    
    # Progress bar
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    # Turn off efficient attention for discriminator (compatibility)
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False
    
    # ============================================================================
    # Compute reference FID statistics (same as original)
    # ============================================================================
    if accelerator.is_main_process and args.track_val_fid:
        write_text(TEXT_FILE, "\nComputing reference FID statistics...\n")
        
        all_ref_images = sorted(os.listdir(os.path.join(args.dataset_folder, "test_B")))
        ref_subset = all_ref_images[: args.num_samples_eval]
        
        feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)
        
        def fn_transform(x):
            x_pil = Image.fromarray(x)
            out_pil = transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.LANCZOS
            )(x_pil)
            return np.array(out_pil)
        
        ref_folder = os.path.join(args.dataset_folder, "test_B")
        
        ref_stats = get_folder_features(
            ref_folder,
            model=feat_model,
            num_workers=0,
            num=len(ref_subset),
            shuffle=False,
            seed=0,
            batch_size=8,
            device=torch.device("cuda"),
            mode="clean",
            custom_image_tranform=fn_transform,
            description="Computing reference FID",
            verbose=True,
        )
        
        write_text(
            TEXT_FILE,
            f"✓ Reference FID stats computed: mu={ref_stats[0].shape}, sigma={ref_stats[1].shape}\n\n"
        )
    
    # ============================================================================
    # Training loop (same as original)
    # ============================================================================
    write_text(TEXT_FILE, "="*80 + "\n")
    write_text(TEXT_FILE, "STARTING TRAINING\n")
    write_text(TEXT_FILE, "="*80 + "\n\n")
    
    global_step = 0
    train_loss_buffer = {"l2": [], "lpips": [], "clipsim": [], "G": [], "D": []}
    
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_pix2pix, net_disc]
            
            if global_step >= args.max_train_steps:
                break
            
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                B, C, H, W = x_src.shape
                
                # ================================================================
                # Forward pass through generator
                # ================================================================
                x_tgt_pred = net_pix2pix(
                    x_src, prompt_tokens=batch["input_ids"], deterministic=True
                )
                
                # ================================================================
                # DIFFERENCE: Standard L2 loss (not weighted)
                # Weighted loss doesn't help with domain displacement
                # ================================================================
                loss_l2 = F.mse_loss(x_tgt_pred, x_tgt) * args.lambda_l2
                
                # LPIPS perceptual loss (same as original)
                loss_lpips = (
                    net_lpips(x_tgt_pred.float(), x_tgt.float()).mean()
                    * args.lambda_lpips
                )
                
                loss = loss_l2 + loss_lpips
                
                # CLIP similarity loss (same as original)
                if args.lambda_clipsim > 0:
                    x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred * 0.5 + 0.5)
                    x_tgt_pred_renorm = F.interpolate(
                        x_tgt_pred_renorm,
                        (224, 224),
                        mode="bilinear",
                        align_corners=False,
                    )
                    caption_tokens = clip.tokenize(batch["caption"], truncate=True).to(
                        x_tgt_pred.device
                    )
                    clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                    loss_clipsim = 1 - clipsim.mean() / 100
                    loss += loss_clipsim * args.lambda_clipsim
                
                # Backprop reconstruction loss
                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                
                # ================================================================
                # Generator adversarial loss (same as original)
                # ================================================================
                x_tgt_pred = net_pix2pix(
                    x_src, prompt_tokens=batch["input_ids"], deterministic=True
                )
                lossG = net_disc(x_tgt_pred, for_G=True).mean() * args.lambda_gan
                
                accelerator.backward(lossG)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                
                # ================================================================
                # Discriminator loss (same as original)
                # ================================================================
                # Real images
                lossD_real = (
                    net_disc(x_tgt.detach(), for_real=True).mean() * args.lambda_gan
                )
                accelerator.backward(lossD_real.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        net_disc.parameters(), args.max_grad_norm
                    )
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                
                # Fake images
                lossD_fake = (
                    net_disc(x_tgt_pred.detach(), for_real=False).mean()
                    * args.lambda_gan
                )
                accelerator.backward(lossD_fake.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        net_disc.parameters(), args.max_grad_norm
                    )
                optimizer_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                
                lossD = lossD_real + lossD_fake
            
            # ================================================================
            # Logging and checkpointing (same as original)
            # ================================================================
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if accelerator.is_main_process:
                    logs = {}
                    logs["lossG"] = lossG.detach().item()
                    logs["lossD"] = lossD.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    if args.lambda_clipsim > 0:
                        logs["loss_clipsim"] = loss_clipsim.detach().item()
                    progress_bar.set_postfix(**logs)
                    
                    # Buffer losses for averaging
                    train_loss_buffer["l2"].append(loss_l2.detach().item())
                    train_loss_buffer["lpips"].append(loss_lpips.detach().item())
                    if args.lambda_clipsim > 0:
                        train_loss_buffer["clipsim"].append(
                            loss_clipsim.detach().item()
                        )
                    train_loss_buffer["G"].append(lossG.detach().item())
                    train_loss_buffer["D"].append(lossD.detach().item())
                    
                    # Save checkpoint
                    if global_step % args.checkpointing_steps == 1 and global_step != 1:
                        outf = os.path.join(
                            args.output_dir,
                            "checkpoints",
                            f"model_step_{global_step}.pkl",
                        )
                        accelerator.unwrap_model(net_pix2pix).save_model(outf)
                    
                    # ========================================================
                    # Validation (same as original)
                    # ========================================================
                    if global_step % args.eval_freq == 1:
                        l_l2, l_lpips, l_clipsim = [], [], []
                        
                        if args.track_val_fid:
                            eval_folder = os.path.join(
                                args.output_dir, "eval", f"fid_{global_step}"
                            )
                            os.makedirs(eval_folder, exist_ok=True)
                        
                        num_eval_samples = 0
                        for step_val, batch_val in enumerate(dl_val):
                            if step_val >= args.num_samples_eval:
                                break
                            
                            x_src = batch_val["conditioning_pixel_values"].cuda()
                            x_tgt = batch_val["output_pixel_values"].cuda()
                            B, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            
                            with torch.no_grad():
                                x_tgt_pred = accelerator.unwrap_model(net_pix2pix)(
                                    x_src,
                                    prompt_tokens=batch_val["input_ids"].cuda(),
                                    deterministic=True,
                                )
                                
                                loss_l2 = F.mse_loss(
                                    x_tgt_pred.float(), x_tgt.float(), reduction="mean"
                                )
                                loss_lpips = net_lpips(
                                    x_tgt_pred.float(), x_tgt.float()
                                ).mean()
                                
                                x_tgt_pred_renorm = t_clip_renorm(
                                    x_tgt_pred * 0.5 + 0.5
                                )
                                x_tgt_pred_renorm = F.interpolate(
                                    x_tgt_pred_renorm,
                                    (224, 224),
                                    mode="bilinear",
                                    align_corners=False,
                                )
                                caption_tokens = clip.tokenize(
                                    batch_val["caption"], truncate=True
                                ).to(x_tgt_pred.device)
                                clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                                clipsim = clipsim.mean()
                                
                                l_l2.append(loss_l2.item())
                                l_lpips.append(loss_lpips.item())
                                l_clipsim.append(clipsim.item())
                                
                                # Save debug output
                                debug_dir = os.path.join(
                                    args.output_dir, "debug_train_outputs"
                                )
                                save_debug_output(
                                    batch_val["conditioning_pixel_values"][0]
                                    .detach()
                                    .cpu(),
                                    x_tgt_pred[0].detach().cpu(),
                                    debug_dir,
                                    global_step,
                                )
                            
                            # Save for FID evaluation
                            if args.track_val_fid:
                                output_pil = transforms.ToPILImage()(
                                    x_tgt_pred[0].cpu() * 0.5 + 0.5
                                )
                                orig_name = os.path.splitext(
                                    batch_val["image_name"][0]
                                )[0]
                                outf = os.path.join(
                                    eval_folder,
                                    f"{orig_name}_fid_{global_step}.png",
                                )
                                output_pil.save(outf)
                                num_eval_samples += 1
                        
                        # Compute FID
                        if args.track_val_fid:
                            curr_stats = get_folder_features(
                                eval_folder,
                                model=feat_model,
                                num_workers=0,
                                num=None,
                                shuffle=False,
                                seed=0,
                                batch_size=8,
                                device=torch.device("cuda"),
                                mode="clean",
                                custom_image_tranform=fn_transform,
                                description="Computing FID",
                                verbose=True,
                            )
                            
                            fid_score = fid_from_feats(ref_stats, curr_stats)
                            logs["val/clean_fid"] = fid_score
                        
                        logs["val/l2"] = np.mean(l_l2)
                        logs["val/lpips"] = np.mean(l_lpips)
                        logs["val/clipsim"] = np.mean(l_clipsim)
                        
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                        # Average training losses
                        if len(train_loss_buffer["l2"]) > 0:
                            logs["train_avg_l2"] = sum(train_loss_buffer["l2"]) / len(
                                train_loss_buffer["l2"]
                            )
                            logs["train_avg_lpips"] = sum(train_loss_buffer["lpips"]) / len(
                                train_loss_buffer["lpips"]
                            )
                            logs["train_avg_G"] = sum(train_loss_buffer["G"]) / len(
                                train_loss_buffer["G"]
                            )
                            logs["train_avg_D"] = sum(train_loss_buffer["D"]) / len(
                                train_loss_buffer["D"]
                            )
                            if args.lambda_clipsim > 0:
                                logs["train_avg_clipsim"] = sum(
                                    train_loss_buffer["clipsim"]
                                ) / len(train_loss_buffer["clipsim"])
                            
                            # Reset buffers
                            for k in train_loss_buffer:
                                train_loss_buffer[k].clear()
                        
                        write_json_metrics(JSON_FILE, global_step, logs, args)
                        write_text(TEXT_FILE, f"Step {global_step}: metrics saved\n")
                    
                    accelerator.log(logs, step=global_step)
    
    write_text(TEXT_FILE, "\n" + "="*80 + "\n")
    write_text(TEXT_FILE, "TRAINING COMPLETE!\n")
    write_text(TEXT_FILE, "="*80 + "\n")


# ============================================================================
# Helper functions (same as original)
# ============================================================================

def write_json_metrics(file_path, step, logs, args):
    """Write metrics to JSON, filtering by active loss weights."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    weight_by_key = {
        "loss_l2": args.lambda_l2,
        "loss_lpips": args.lambda_lpips,
        "loss_clipsim": args.lambda_clipsim,
        "lossG": args.lambda_gan,
        "lossD": args.lambda_gan,
        "val/l2": args.lambda_l2,
        "val/lpips": args.lambda_lpips,
        "val/clipsim": args.lambda_clipsim,
    }
    
    simple_logs = {}
    for k, v in logs.items():
        if k in weight_by_key and weight_by_key[k] <= 0:
            continue
        
        if isinstance(v, (int, float, str, bool)):
            simple_logs[k] = float(v) if isinstance(v, (int, float)) else v
    
    simple_logs["step"] = int(step)
    
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(simple_logs) + "\n")


def save_debug_batch(batch, outdir, prefix, max_images=8):
    """Save conditioning and target images from a batch."""
    os.makedirs(outdir, exist_ok=True)
    
    x_src = batch["conditioning_pixel_values"]
    x_tgt = batch["output_pixel_values"]
    x_tgt_vis = (x_tgt * 0.5 + 0.5).clamp(0.0, 1.0)
    
    B = x_src.size(0)
    n = min(max_images, B)
    
    for i in range(n):
        vutils.save_image(x_src[i], os.path.join(outdir, f"{prefix}_A_{i}.png"))
        vutils.save_image(x_tgt_vis[i], os.path.join(outdir, f"{prefix}_B_{i}.png"))


def save_debug_output(x_src, x_pred, outdir, step):
    """Save a single model output for debugging."""
    os.makedirs(outdir, exist_ok=True)
    
    x_pred_vis = (x_pred * 0.5 + 0.5).clamp(0, 1)
    
    vutils.save_image(x_src, os.path.join(outdir, f"step_{step}_A.png"))
    vutils.save_image(x_pred_vis, os.path.join(outdir, f"step_{step}_B.png"))


def write_text(output_file, text):
    """Append text to log file."""
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(str(text))


if __name__ == "__main__":
    args = parse_args_paired_training()
    
    # DIFFERENCE: Require cyclegan_checkpoint argument
    if not hasattr(args, 'cyclegan_checkpoint') or args.cyclegan_checkpoint is None:
        raise ValueError(
            "Must provide --cyclegan_checkpoint argument with path to CycleGAN .pkl file"
        )
    
    main(args)


"""
Example usage:

# Fine-tune with per-image semantic prompts (uses train_prompts.json)
accelerate launch src/train_pix2pix_turbo_from_cyclegan.py \
    --cyclegan_checkpoint="outputs/cyclegan_run/checkpoints/model_5000.pkl" \
    --output_dir="outputs/finetuned_with_prompts/" \
    --dataset_folder="/data/upftfg19/mfsvensson/Data_TFG/myPairedDataset" \
    --data_set_type="original" \
    --resolution=512 \
    --train_batch_size=16 \
    --enable_xformers_memory_efficient_attention \
    --track_val_fid \
    --dataloader_num_workers=4 \
    --mixed_precision="bf16" \
    --max_train_steps=10000 \
    --checkpointing_steps=500 \
    --eval_freq=500

# Fine-tune with generic transformation prompt
accelerate launch src/train_pix2pix_turbo_from_cyclegan.py \
    --cyclegan_checkpoint="outputs/cyclegan_run/checkpoints/model_5000.pkl" \
    --output_dir="outputs/finetuned_generic/" \
    --dataset_folder="/data/upftfg19/mfsvensson/Data_TFG/myPairedDataset" \
    --data_set_type="modified" \
    --resolution=512 \
    --train_batch_size=16 \
    --enable_xformers_memory_efficient_attention \
    --track_val_fid \
    --dataloader_num_workers=4 \
    --mixed_precision="bf16" \
    --max_train_steps=10000 \
    --checkpointing_steps=500 \
    --eval_freq=500
"""
