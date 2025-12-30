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
# import wandb
from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats

from pix2pix_turbo import Pix2Pix_Turbo
from my_utils.training_utils import parse_args_paired_training, PairedDataset

def main(args):
    # create one text file and one json file at the output dir 
    os.makedirs(args.output_dir, exist_ok=True)
    JSON_FILE = os.path.join(args.output_dir, "metrics.jsonl")
    metrics_file = JSON_FILE
    TEXT_FILE = os.path.join(args.output_dir, "runing_information.txt")
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to, # Passed as a flag when executing the script, accelerate will configure wandb tracking
    )
    write_text(TEXT_FILE, "WE HAVE STARTED!\n\n")
    write_text(TEXT_FILE, "\nAccelerator created: mixed_precision={accelerator.mixed_precision}, num_processes={accelerator.num_processes}, local_main={accelerator.is_local_main_process}\n")

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    # No need for if-condition here 
    net_pix2pix = Pix2Pix_Turbo(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae, ignore_skip=args.ignore_skip, skip_weight=args.skip_weight)
    net_pix2pix.set_train()
    write_text(TEXT_FILE, f"Initalized well the UNet\n")
    write_text(TEXT_FILE, f"net_pix2pix created; requires_grad counts: total_params={sum(p.numel() for p in net_pix2pix.parameters())}, trainable_params={sum(p.numel() for p in net_pix2pix.parameters() if p.requires_grad)}")
    # Enable xformers memory efficient attention
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_pix2pix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_pix2pix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Discriminator based loss adds semantic quality to the outputs
    # Trying to preserve fine details and perceptual quality during image translation, even though they have paired data
    if args.gan_disc_type == "vagan_clip":
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
    else:
        raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")

    # prepare the models by brining them to GPU and setting training modes
    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    write_text(TEXT_FILE, "\nPASSED THE VGG TEST\n")
    net_clip, _ = clip.load("ViT-B/32", device="cuda")
    # this is to freeze the CLIP model
    net_clip.requires_grad_(False)
    net_clip.eval()

    net_lpips.requires_grad_(False)

    # Pass ONLY those parameters to optimizer that need to be updated
    # This is where the layers to be updated are specified
    # Performance: Slightly faster since optimizer doesn't iterate over frozen parameters
    layers_to_opt = []
    for n, _p in net_pix2pix.unet.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt += list(net_pix2pix.unet.conv_in.parameters())
    for n, _p in net_pix2pix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    # add the skip conv layers for decoder
    if not args.ignore_skip:
        layers_to_opt = layers_to_opt + list(net_pix2pix.vae.decoder.skip_conv_1.parameters()) + \
            list(net_pix2pix.vae.decoder.skip_conv_2.parameters()) + \
            list(net_pix2pix.vae.decoder.skip_conv_3.parameters()) + \
            list(net_pix2pix.vae.decoder.skip_conv_4.parameters())

    write_text(TEXT_FILE, f"\nNumber of parameters to optimize: {sum(p.numel() for p in layers_to_opt)}\n")
    # Build the list of parameters that will be optimized. We explicitly collect
    # only layers that should be updated (LoRA layers + a few conv layers here).
    # This avoids passing frozen parameters to the optimizer which (a) saves
    # memory and (b) makes optimizer.step() slightly faster because it doesn't
    # iterate over parameters that won't change.

    # Create an AdamW optimizer for the generator parameters we selected.
    # - `lr`, `betas`, `weight_decay` and `eps` are taken from CLI args.
    # - We use AdamW (Adam with decoupled weight decay) which is common for
    #   transformer and diffusion style models.
    optimizer = torch.optim.AdamW(
        layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # The learning-rate scheduler is created with `get_scheduler` from
    # `diffusers.optimization`. Note two important points:
    # 1) We pass the optimizer instance so the scheduler can update its LR.
    # 2) `num_warmup_steps` and `num_training_steps` are multiplied by
    #    `accelerator.num_processes`. This is because the script assumes the
    #    total number of optimization steps scales with the number of
    #    parallel processes (for multi-GPU/distributed runs). `Accelerate`
    #    sometimes requires scaling these counts so the warmup and total-step
    #    schedules line up across processes.
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # The discriminator uses a separate optimizer and scheduler. Here we pass
    # `net_disc.parameters()` (all discriminator parameters since the
    # discriminator is trained normally) to another AdamW instance.
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
    # Prepare the training and validation datasets.
    dataset_train = PairedDataset(dataset_folder=args.dataset_folder, image_prep=args.train_image_prep, 
    split="train", tokenizer=net_pix2pix.tokenizer, data_set_type=args.data_set_type,
    use_canny_conditioning=args.use_canny_conditioning, canny_low_threshold=args.canny_low_threshold,
    canny_high_threshold=args.canny_high_threshold, binary_threshold=args.binary_threshold)
    # DataLoader calls the methods __len__ and __getitem__ on the dataset, these come from the custom PairedDataset class
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dataset_val = PairedDataset(dataset_folder=args.dataset_folder, image_prep=args.test_image_prep, 
    split="test", tokenizer=net_pix2pix.tokenizer, data_set_type=args.data_set_type,use_canny_conditioning=args.use_canny_conditioning, canny_low_threshold=args.canny_low_threshold,
    canny_high_threshold=args.canny_high_threshold, binary_threshold=args.binary_threshold)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)
    write_text(TEXT_FILE, f"\nDataLoaders created: train size={len(dataset_train)}, val size={len(dataset_val)}\n")
    
    # ===== DEBUG: save a few samples from train_A/B and test_A/B after DataLoader =====
    debug_dir = os.path.join(args.output_dir, "debug_dataloader")
    print(f"[DEBUG] Saving a few samples from dataloaders to: {debug_dir}")

    # One batch from train
    batch_train = next(iter(dl_train))
    save_debug_batch(batch_train, debug_dir, prefix="train", max_images=20)
    batch_train = next(iter(dl_train))
    save_debug_batch(batch_train, debug_dir, prefix="train", max_images=20)

    # One batch from val/test
    batch_val = next(iter(dl_val))
    save_debug_batch(batch_val, debug_dir, prefix="test", max_images=10)
    batch_val = next(iter(dl_val))
    save_debug_batch(batch_val, debug_dir, prefix="test", max_images=10)
    # ===== END DEBUG =====



    # Prepare everything with our `accelerator`.
    net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
        net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc
    )
    net_clip, net_lpips = accelerator.prepare(net_clip, net_lpips)
    # renorm with image net statistics
    t_clip_renorm = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move all networks to weight_dtype # device is set automatically in accelerator.prepare
    net_pix2pix.to(dtype=weight_dtype)
    net_disc.to(dtype=weight_dtype)
    net_lpips.to(dtype=weight_dtype)
    net_clip.to(dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # turn off eff. attn for the discriminator
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False
    
    # Get examples for ref FID
    all_ref_images = sorted(os.listdir(os.path.join(args.dataset_folder, "test_B")))
    ref_subset = all_ref_images[:args.num_samples_eval]  # 100 por defecto
    # compute the reference stats for FID tracking
    # compute the reference stats for FID tracking
    if accelerator.is_main_process and args.track_val_fid:
        write_text(TEXT_FILE, "Computing reference stats for FID...\n")
        feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)
        write_text(TEXT_FILE, "PASSED THE build_feature_extractor TEST\n")
        
        def fn_transform(x):
            x_pil = Image.fromarray(x)
            out_pil = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.LANCZOS)(x_pil)
            return np.array(out_pil)
        
        ref_folder = os.path.join(args.dataset_folder, "test_B")
        write_text(TEXT_FILE, f"Reference folder: {ref_folder}\n")
        
        ref_stats = get_folder_features(ref_folder, model=feat_model, num_workers=0, num=len(ref_subset),
                shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                mode="clean", custom_image_tranform=fn_transform, description="", verbose=True)
        
        write_text(TEXT_FILE, f"REF_STATS: mu shape={ref_stats[0].shape}, sigma shape={ref_stats[1].shape}\n")
    write_text(TEXT_FILE, f"{ref_stats}")
    # start the training loop
    global_step = 0
    # to calcualte the average loss every 100 steps 
    train_loss_buffer = {
    "l2": [],
    "lpips": [],
    "clipsim": [],
    "G": [],
    "D": []
    }
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_pix2pix, net_disc]
            if global_step >= args.max_train_steps:
                break
            if args.print_pure_model:
                outf = os.path.join(args.output_dir, "checkpoints", f"untouched_naked_model.pkl")
                accelerator.unwrap_model(net_pix2pix).save_model(outf)
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                B, C, H, W = x_src.shape
                # forward pass
                x_tgt_pred = net_pix2pix(x_src, prompt_tokens=batch["input_ids"], deterministic=True)
                # Reconstruction loss
                with torch.no_grad():
                    gray = x_tgt.mean(dim=1, keepdim=True)      # [B,1,H,W], still [-1,1]
                    ink_mask = (gray < 0.5).float()             # in the range, -1 is completly black and 1 is white 
                    w_bg, w_fg = 0.5, 2.0
                    weights = w_bg + (w_fg - w_bg) * ink_mask   # 0.5 for bg, 2.0 for lines (all black dots at 1 and white are 0 in the mask)
                    weights = weights / weights.mean()          # keep average weight ≈ 1
                # Here diff2 is an array of the pixel losses 
                diff2 = (x_tgt_pred - x_tgt) ** 2
                # Here we do the weighted to prioritze the lines
                loss_l2 = (diff2 * weights).mean()*args.lambda_l2

                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips
                loss = loss_l2 + loss_lpips
                # CLIP similarity loss
                if args.lambda_clipsim > 0:
                    x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred * 0.5 + 0.5)
                    x_tgt_pred_renorm = F.interpolate(x_tgt_pred_renorm, (224, 224), mode="bilinear", align_corners=False)
                    caption_tokens = clip.tokenize(batch["caption"], truncate=True).to(x_tgt_pred.device)
                    clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                    loss_clipsim = (1 - clipsim.mean() / 100)
                    loss += loss_clipsim * args.lambda_clipsim
                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                """
                Generator loss: fool the discriminator
                """
                x_tgt_pred = net_pix2pix(x_src, prompt_tokens=batch["input_ids"], deterministic=True)
                lossG = net_disc(x_tgt_pred, for_G=True).mean() * args.lambda_gan
                accelerator.backward(lossG)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                """
                Discriminator loss: fake image vs real image
                """
                # real image
                lossD_real = net_disc(x_tgt.detach(), for_real=True).mean() * args.lambda_gan
                accelerator.backward(lossD_real.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                # fake image
                lossD_fake = net_disc(x_tgt_pred.detach(), for_real=False).mean() * args.lambda_gan
                accelerator.backward(lossD_fake.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                lossD = lossD_real + lossD_fake

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    logs["lossG"] = lossG.detach().item()
                    logs["lossD"] = lossD.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    if args.lambda_clipsim > 0:
                        logs["loss_clipsim"] = loss_clipsim.detach().item()
                    progress_bar.set_postfix(**logs)

                    # add the loss to the list
                    train_loss_buffer["l2"].append(loss_l2.detach().item())
                    train_loss_buffer["lpips"].append(loss_lpips.detach().item())
                    if args.lambda_clipsim > 0:
                        train_loss_buffer["clipsim"].append(loss_clipsim.detach().item())
                    train_loss_buffer["G"].append(lossG.detach().item())
                    train_loss_buffer["D"].append(lossD.detach().item())


                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1 and global_step != 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_model_Fill50k_{global_step}.pkl")
                        accelerator.unwrap_model(net_pix2pix).save_model(outf)
                    
                    # compute validation set FID, L2, LPIPS, CLIP-SIM
                    # compute validation set FID, L2, LPIPS, CLIP-SIM
                    if global_step % args.eval_freq == 1:
                        l_l2, l_lpips, l_clipsim = [], [], []
                        if args.track_val_fid:
                            eval_folder = os.path.join(args.output_dir, "eval", f"fid_{global_step}")
                            os.makedirs(eval_folder, exist_ok=True)
                        
                        num_eval_samples = 0
                        for step, batch_val in enumerate(dl_val):
                            if step >= args.num_samples_eval:
                                break
                            x_src = batch_val["conditioning_pixel_values"].cuda()
                            x_tgt = batch_val["output_pixel_values"].cuda()
                            B, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            
                            with torch.no_grad():
                                x_tgt_pred = accelerator.unwrap_model(net_pix2pix)(x_src, prompt_tokens=batch_val["input_ids"].cuda(), deterministic=True)
                                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean")
                                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean()
                                
                                x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred * 0.5 + 0.5)
                                x_tgt_pred_renorm = F.interpolate(x_tgt_pred_renorm, (224, 224), mode="bilinear", align_corners=False)
                                caption_tokens = clip.tokenize(batch_val["caption"], truncate=True).to(x_tgt_pred.device)
                                clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                                clipsim = clipsim.mean()
                                
                                l_l2.append(loss_l2.item())
                                l_lpips.append(loss_lpips.item())
                                l_clipsim.append(clipsim.item())
                                # Before each just print one result from runing the network to see (outside of FID scope)
                                debug_dir = os.path.join(args.output_dir, "debug_train_outputs")
                                # pick sample 0 from batch
                                save_debug_output(
                                    batch_val["conditioning_pixel_values"][0].detach().cpu(),
                                    x_tgt_pred[0].detach().cpu(),
                                    debug_dir,
                                    global_step
                                )
                            
                            # save output images to file for FID evaluation
                            if args.track_val_fid:
                                # For saving, they do x_tgt_pred * 0.5 + 0.5 ⇒ this maps [-1, 1] → [0, 1].
                                output_pil = transforms.ToPILImage()(x_tgt_pred[0].cpu() * 0.5 + 0.5)
                                orig_name = os.path.splitext(batch_val["image_name"][0])[0]
                                outf = os.path.join(
                                    args.output_dir,
                                    "eval",
                                    f"fid_{global_step}",
                                    f"{orig_name}_fid_{global_step}.png")
                                output_pil.save(outf)
                                num_eval_samples += 1
                        
                        #write_text(TEXT_FILE, f"Eval step {global_step}: Generated {num_eval_samples} images\n")
                        
                        if args.track_val_fid:
                            #write_text(TEXT_FILE, f"Computing FID from folder: {eval_folder}\n")
                            curr_stats = get_folder_features(eval_folder, model=feat_model, num_workers=0, num=None,
                                    shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                                    mode="clean", custom_image_tranform=fn_transform, description="", verbose=True)
                            
                            #write_text(TEXT_FILE, f"CURR_STATS: mu shape={curr_stats[0].shape}, sigma shape={curr_stats[1].shape}\n")
                            #write_text(TEXT_FILE, f"COMPARING: ref_stats mu={ref_stats[0].shape} vs curr_stats mu={curr_stats[0].shape}\n")
                            #write_text(TEXT_FILE, f"COMPARING: ref_stats sigma={ref_stats[1].shape} vs curr_stats sigma={curr_stats[1].shape}\n")
                            
                            fid_score = fid_from_feats(ref_stats, curr_stats)
                            logs["val/clean_fid"] = fid_score
                        logs["val/l2"] = np.mean(l_l2)
                        logs["val/lpips"] = np.mean(l_lpips)
                        logs["val/clipsim"] = np.mean(l_clipsim)
                        gc.collect()
                        torch.cuda.empty_cache()
                        # Save the parameters 
                        # Calculate the averages
                        # ---- ADD THIS INSIDE THE eval_freq BLOCK, BEFORE validation starts ----
                        if len(train_loss_buffer["l2"]) > 0:
                            avg_train_l2     = sum(train_loss_buffer["l2"]) / len(train_loss_buffer["l2"])
                            avg_train_lpips  = sum(train_loss_buffer["lpips"]) / len(train_loss_buffer["lpips"])
                            avg_train_G      = sum(train_loss_buffer["G"]) / len(train_loss_buffer["G"])
                            avg_train_D      = sum(train_loss_buffer["D"]) / len(train_loss_buffer["D"])
                            if args.lambda_clipsim > 0:
                                avg_train_clipsim = sum(train_loss_buffer["clipsim"]) / len(train_loss_buffer["clipsim"])

                            # Log the averaged training losses
                            logs["train_avg_l2"]     = avg_train_l2
                            logs["train_avg_lpips"]  = avg_train_lpips
                            logs["train_avg_G"]      = avg_train_G
                            logs["train_avg_D"]      = avg_train_D
                            if args.lambda_clipsim > 0:
                                logs["train_avg_clipsim"] = avg_train_clipsim

                            # Reset for next block
                            for k in train_loss_buffer:
                                train_loss_buffer[k].clear()

                        write_json_metrics(JSON_FILE, global_step, logs, args)
                        write_text(TEXT_FILE, f"Step {global_step}: wrote metrics")

                    accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
                
                    
"""
def write_json_metrics(file_path, step, logs):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    simple_logs = {}
    for k, v in logs.items():
        if isinstance(v, (int, float, str, bool)):
            simple_logs[k] = float(v) if isinstance(v, (int, float)) else v

    simple_logs["step"] = int(step)

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(simple_logs) + "\n")
"""

def write_json_metrics(file_path, step, logs, args):
    """
    Write metrics to JSON, but only keep loss terms whose corresponding λ > 0.
    Non-loss metrics (e.g. FID) are always kept.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Map metric names -> their associated weight in args
    weight_by_key = {
        # generator / reconstruction losses
        "loss_l2":        args.lambda_l2,
        "loss_lpips":     args.lambda_lpips,
        "loss_clipsim":   args.lambda_clipsim,
        "lossG":          args.lambda_gan,
        "lossD":          args.lambda_gan,

        # validation losses
        "val/l2":         args.lambda_l2,
        "val/lpips":      args.lambda_lpips,
        "val/clipsim":    args.lambda_clipsim,
    }

    simple_logs = {}
    for k, v in logs.items():
        # If this metric has an associated λ and it is <= 0, skip it
        if k in weight_by_key and weight_by_key[k] <= 0:
            continue

        if isinstance(v, (int, float, str, bool)):
            simple_logs[k] = float(v) if isinstance(v, (int, float)) else v

    simple_logs["step"] = int(step)

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(simple_logs) + "\n")

    

def save_debug_batch(batch, outdir, prefix, max_images=8):
    """
    Save a few conditioning (A) and target (B) images from a dataloader batch
    to visually verify that the dataset + transforms are correct.
    """
    os.makedirs(outdir, exist_ok=True)
    x_src = batch["conditioning_pixel_values"]      # [B, C, H, W], in [0, 1]
    x_tgt = batch["output_pixel_values"]           # [B, C, H, W], in [-1, 1] because of normalize(mean=0.5, std=0.5)

    # De-normalize targets back to [0, 1] for saving
    x_tgt_vis = (x_tgt * 0.5 + 0.5).clamp(0.0, 1.0)
    B = x_src.size(0)
    n = min(max_images, B)

    for i in range(n):
        vutils.save_image(
            x_src[i],
            os.path.join(outdir, f"{prefix}_A_{i}.png")
        )
        vutils.save_image(
            x_tgt_vis[i],
            os.path.join(outdir, f"{prefix}_B_{i}.png")
        )

def save_debug_output(x_src, x_pred, outdir, step):
    """
    Save a single output image (and optionally its input) for quick visual debugging.
    x_src: conditioning image in [0,1]
    x_pred: model output in [-1,1]
    """
    os.makedirs(outdir, exist_ok=True)

    # De-normalize prediction: [-1,1] -> [0,1]
    x_pred_vis = (x_pred * 0.5 + 0.5).clamp(0, 1)

    # Optional: also save the conditioning input (already in [0,1])
    vutils.save_image(x_src, os.path.join(outdir, f"step_{step}_A.png"))
    vutils.save_image(x_pred_vis, os.path.join(outdir, f"step_{step}_B.png"))


def write_text(output_dir, text):
    """
    Append a line of text to output_dir/log.txt
    """
    with open(output_dir, "a", encoding="utf-8") as f:
        f.write(str(text))

if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)

"""
Command: 

accelerate launch src/train_pix2pix_turbo.py \
    --pretrained_model_name_or_path="/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo" \
    --output_dir="outputs/myPairedDataset/" \
    --dataset_folder="/data/upftfg19/mfsvensson/Data_TFG/myPairedDatasets" \
    --resolution=512 \
    --train_batch_size=32 \
    --enable_xformers_memory_efficient_attention --viz_freq 25 \
    --track_val_fid \
    --dataloader_num_workers 7 \
    --mixed_precision="bf16" \
"""
