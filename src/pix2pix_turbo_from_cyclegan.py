"""
Pix2Pix_Turbo model that loads weights from a trained CycleGAN_Turbo checkpoint.

KEY DIFFERENCES from pix2pix_turbo.py:
1. UNet uses 3 separate LoRA adapters (encoder/decoder/others) instead of 1
2. Loads from CycleGAN checkpoint format with sd_encoder/sd_decoder/sd_other
3. Uses single-direction VAE (a2b only, no bidirectional wrapper)
4. Saves checkpoints in CycleGAN-compatible format for inference compatibility
"""

import os
import sys
import copy
import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig

p = "src/"
sys.path.append(p)
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd


class Pix2Pix_Turbo_from_CycleGAN(torch.nn.Module):
    """
    Pix2Pix model initialized from a CycleGAN checkpoint.
    
    This class is designed to:
    - Load a trained CycleGAN model
    - Fine-tune it on paired data using pix2pix losses
    - Save checkpoints compatible with cyclegan_turbo.py for inference
    """
    
    def __init__(
        self, 
        cyclegan_checkpoint_path,
        lora_rank_unet=8, 
        lora_rank_vae=4,
        ignore_skip=False,
        skip_weight=1.0
    ):
        """
        Initialize model by loading a CycleGAN checkpoint.
        
        Args:
            cyclegan_checkpoint_path: Path to .pkl checkpoint from CycleGAN training
            lora_rank_unet: LoRA rank for UNet (should match checkpoint)
            lora_rank_vae: LoRA rank for VAE (should match checkpoint)
            ignore_skip: Whether to ignore skip connections in VAE decoder
            skip_weight: Weight for skip connections
        """
        super().__init__()
        
        print(f"\n{'='*80}")
        print(f"Initializing Pix2Pix_Turbo_from_CycleGAN")
        print(f"Loading checkpoint: {cyclegan_checkpoint_path}")
        print(f"{'='*80}\n")
        
        # Load tokenizer and text encoder (frozen, used for conditioning)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo", 
            subfolder="tokenizer", 
            local_files_only=True
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            "/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo", 
            subfolder="text_encoder"
        ).cuda()
        self.text_encoder.requires_grad_(False)
        
        # Create scheduler for 1-step diffusion
        self.sched = make_1step_sched()
        
        # Initialize base VAE
        print("Loading base VAE...")
        vae = AutoencoderKL.from_pretrained(
            "/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo", 
            subfolder="vae"
        )
        
        # Replace VAE encoder/decoder forward methods with custom versions
        # These custom versions capture skip connections during encoding
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        
        # DIFFERENCE: Single direction VAE only (a2b: photo -> sticker)
        # CycleGAN has bidirectional VAEs, but for pix2pix we only need one direction
        
        # Add skip connection convolution layers to VAE decoder
        # These allow information to flow from encoder to decoder
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = ignore_skip
        vae.decoder.skip_weight = skip_weight
        vae.decoder.gamma = 1
        
        # Initialize base UNet
        print("Loading base UNet...")
        unet = UNet2DConditionModel.from_pretrained(
            "/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo", 
            subfolder="unet"
        )
        
        # Load the CycleGAN checkpoint and set up LoRA adapters
        print("\nLoading CycleGAN checkpoint...")
        self._load_cyclegan_checkpoint(
            unet, vae, cyclegan_checkpoint_path, lora_rank_unet, lora_rank_vae
        )
        
        # Move to GPU
        unet.to("cuda")
        vae.to("cuda")
        
        self.unet = unet
        self.vae = vae
        self.timesteps = torch.tensor([999], device="cuda").long()
        
        # Store config for saving
        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae
        
        print(f"\n{'='*80}")
        print("Model initialization complete!")
        print(f"{'='*80}\n")
    
    def _load_cyclegan_checkpoint(self, unet, vae, checkpoint_path, rank_unet, rank_vae):
        """
        Load weights from CycleGAN checkpoint into UNet and VAE.
        
        CRITICAL DIFFERENCE: CycleGAN uses 3 separate LoRA adapters for UNet:
        - default_encoder: for down_blocks + conv_in
        - default_decoder: for up_blocks
        - default_others: for mid_block and other modules
        
        The checkpoint stores these as separate state dicts.
        """
        print(f"Reading checkpoint file: {checkpoint_path}")
        sd = torch.load(checkpoint_path, map_location="cpu")
        
        # Verify checkpoint structure
        required_keys = ["sd_encoder", "sd_decoder", "sd_other", "sd_vae_enc", "sd_vae_dec"]
        for key in required_keys:
            if key not in sd:
                raise ValueError(f"Checkpoint missing required key: {key}")
        
        print(f"✓ Checkpoint structure verified")
        print(f"  - UNet encoder modules: {len(sd['l_target_modules_encoder'])}")
        print(f"  - UNet decoder modules: {len(sd['l_target_modules_decoder'])}")
        print(f"  - UNet other modules: {len(sd['l_modules_others'])}")
        print(f"  - VAE target modules: {len(sd['vae_lora_target_modules'])}")
        
        # ============================================================================
        # LOAD UNET - 3 SEPARATE LORA ADAPTERS
        # ============================================================================
        print("\nSetting up UNet LoRA adapters...")
        
        # Create LoRA configs for each adapter
        # These define which modules get LoRA layers and the rank
        lora_conf_encoder = LoraConfig(
            r=rank_unet,
            init_lora_weights="gaussian",
            target_modules=sd["l_target_modules_encoder"],
            lora_alpha=rank_unet
        )
        lora_conf_decoder = LoraConfig(
            r=rank_unet,
            init_lora_weights="gaussian",
            target_modules=sd["l_target_modules_decoder"],
            lora_alpha=rank_unet
        )
        lora_conf_others = LoraConfig(
            r=rank_unet,
            init_lora_weights="gaussian",
            target_modules=sd["l_modules_others"],
            lora_alpha=rank_unet
        )
        
        # Add the three adapters to UNet
        # Adapter names must match CycleGAN format: "default_encoder", "default_decoder", "default_others"
        unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        unet.add_adapter(lora_conf_others, adapter_name="default_others")
        
        print("✓ Created 3 LoRA adapters for UNet")
        
        # Load weights into each adapter
        # Parameter names in the model include the adapter name as a suffix
        # e.g., "down_blocks.0.attentions.0.to_k.default_encoder.weight"
        # We need to match these to the saved state dict keys which don't have the suffix
        
        print("Loading encoder weights...")
        encoder_count = 0
        for n, p in unet.named_parameters():
            if "lora" in n and "default_encoder" in n:
                # Remove the adapter suffix to get the key in saved state dict
                name_sd = n.replace(".default_encoder.weight", ".weight")
                if name_sd in sd["sd_encoder"]:
                    p.data.copy_(sd["sd_encoder"][name_sd])
                    encoder_count += 1
        print(f"✓ Loaded {encoder_count} encoder LoRA weights")
        
        print("Loading decoder weights...")
        decoder_count = 0
        for n, p in unet.named_parameters():
            if "lora" in n and "default_decoder" in n:
                name_sd = n.replace(".default_decoder.weight", ".weight")
                if name_sd in sd["sd_decoder"]:
                    p.data.copy_(sd["sd_decoder"][name_sd])
                    decoder_count += 1
        print(f"✓ Loaded {decoder_count} decoder LoRA weights")
        
        print("Loading other module weights...")
        others_count = 0
        for n, p in unet.named_parameters():
            if "lora" in n and "default_others" in n:
                name_sd = n.replace(".default_others.weight", ".weight")
                if name_sd in sd["sd_other"]:
                    p.data.copy_(sd["sd_other"][name_sd])
                    others_count += 1
        print(f"✓ Loaded {others_count} other LoRA weights")
        
        # Activate all three adapters
        unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
        
        # ============================================================================
        # LOAD VAE - SINGLE DIRECTION (a2b only)
        # ============================================================================
        print("\nSetting up VAE LoRA adapter...")
        
        # Create VAE LoRA config
        vae_lora_config = LoraConfig(
            r=rank_vae,
            init_lora_weights="gaussian",
            target_modules=sd["vae_lora_target_modules"]
        )
        vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        print("✓ Created LoRA adapter for VAE")
        
        # DIFFERENCE: CycleGAN saves VAE weights in wrapped format (VAE_encode/VAE_decode)
        # We need to extract only the a2b direction (vae.) and ignore b2a (vae_b2a.)
        
        print("Loading VAE weights (a2b direction only)...")
        
        # The sd_vae_enc and sd_vae_dec contain weights for both directions
        # Keys are like: "vae.decoder.skip_conv_1.weight" (a2b) and "vae_b2a.decoder.skip_conv_1.weight" (b2a)
        # We only want the "vae." keys
        
        vae_state_dict = vae.state_dict()
        vae_loaded_count = 0
        
        # Load from encoder wrapper state dict
        for key, value in sd["sd_vae_enc"].items():
            # Only load a2b direction (keys starting with "vae.")
            if key.startswith("vae.") and not key.startswith("vae_b2a."):
                clean_key = key.replace("vae.", "")  # Remove "vae." prefix
                if clean_key in vae_state_dict:
                    vae_state_dict[clean_key] = value
                    vae_loaded_count += 1
        
        # Load from decoder wrapper state dict
        for key, value in sd["sd_vae_dec"].items():
            # Only load a2b direction
            if key.startswith("vae.") and not key.startswith("vae_b2a."):
                clean_key = key.replace("vae.", "")
                if clean_key in vae_state_dict:
                    vae_state_dict[clean_key] = value
                    vae_loaded_count += 1
        
        # Load the state dict into VAE
        vae.load_state_dict(vae_state_dict, strict=False)
        print(f"✓ Loaded {vae_loaded_count} VAE weights")
        
        # Verify skip connection layers were loaded correctly
        print("\n" + "="*80)
        print("VAE Skip Connection Verification:")
        print("="*80)
        skip_layers_ok = True
        for name, param in vae.named_parameters():
            if 'skip_conv' in name.lower():
                param_sum = param.abs().sum().item()
                param_mean = param.abs().mean().item()
                status = "✓ OK" if param_sum > 1e-6 else "✗ WARNING: Near zero!"
                print(f"{status} {name:50s} | sum={param_sum:12.6f} | mean={param_mean:12.6f}")
                if param_sum < 1e-6:
                    skip_layers_ok = False
        
        if not skip_layers_ok:
            print("\n⚠ WARNING: Some skip layers appear to be near zero!")
            print("This may indicate a loading issue.")
        else:
            print("\n✓ All skip layers loaded successfully!")
        print("="*80)
        
        # Store target modules for saving later
        self.l_target_modules_encoder = sd["l_target_modules_encoder"]
        self.l_target_modules_decoder = sd["l_target_modules_decoder"]
        self.l_modules_others = sd["l_modules_others"]
        self.vae_lora_target_modules = sd["vae_lora_target_modules"]
    
    def set_train(self):
        """
        Set model to training mode and configure which parameters require gradients.
        Same as pix2pix_turbo.py
        """
        self.unet.train()
        self.vae.train()
        
        # Enable gradients for UNet LoRA layers
        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.requires_grad = True
        
        # Enable gradients for UNet conv_in (first convolution)
        self.unet.conv_in.requires_grad_(True)
        
        # Enable gradients for VAE LoRA layers
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.requires_grad = True
        
        # Enable gradients for VAE skip connection layers
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)
    
    def set_eval(self):
        """Set model to evaluation mode."""
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
    
    def forward(self, c_t, prompt=None, prompt_tokens=None, deterministic=True):
        """
        Forward pass through the model.
        Same as pix2pix_turbo.py - single direction, deterministic generation.
        
        Args:
            c_t: Conditioning image (input) in range [-1, 1]
            prompt: Text prompt string (optional)
            prompt_tokens: Pre-tokenized prompt (optional)
            deterministic: Whether to use deterministic generation (always True for pix2pix)
        
        Returns:
            Generated image in range [-1, 1]
        """
        # Either prompt or prompt_tokens must be provided
        assert (prompt is None) != (prompt_tokens is None), \
            "Either prompt or prompt_tokens should be provided"
        
        if prompt is not None:
            # Tokenize and encode text prompt
            caption_tokens = self.tokenizer(
                prompt, 
                max_length=self.tokenizer.model_max_length,
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            ).input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            # Use pre-tokenized prompt
            caption_enc = self.text_encoder(prompt_tokens)[0]
        
        # Encode input image to latent space
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        
        # Run UNet to predict the denoised latent
        model_pred = self.unet(
            encoded_control, 
            self.timesteps, 
            encoder_hidden_states=caption_enc
        ).sample
        
        # Apply scheduler step (1-step diffusion)
        x_denoised = self.sched.step(
            model_pred, 
            self.timesteps, 
            encoded_control, 
            return_dict=True
        ).prev_sample
        x_denoised = x_denoised.to(model_pred.dtype)
        
        # Decode latent to image, using skip connections from encoder
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample
        output_image = output_image.clamp(-1, 1)
        
        return output_image
    
    def save_model(self, outf):
        """
        Save model checkpoint in CycleGAN-compatible format.
        
        DIFFERENCE: Saves UNet weights in 3 separate state dicts (encoder/decoder/others)
        This allows the checkpoint to be loaded by cyclegan_turbo.py for inference.
        
        Args:
            outf: Output file path (.pkl)
        """
        print(f"\nSaving model checkpoint to: {outf}")
        
        sd = {}
        
        # Save LoRA configuration
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["l_target_modules_encoder"] = self.l_target_modules_encoder
        sd["l_target_modules_decoder"] = self.l_target_modules_decoder
        sd["l_modules_others"] = self.l_modules_others
        sd["vae_lora_target_modules"] = self.vae_lora_target_modules
        
        # CRITICAL: Save UNet weights in 3 separate dictionaries
        # This matches CycleGAN checkpoint format
        
        unet_state = self.unet.state_dict()
        
        sd["sd_encoder"] = {}
        sd["sd_decoder"] = {}
        sd["sd_other"] = {}
        
        # Split UNet weights by adapter
        for k, v in unet_state.items():
            if "lora" in k or "conv_in" in k:
                if "default_encoder" in k:
                    # Remove adapter suffix for storage
                    clean_key = k.replace(".default_encoder.weight", ".weight")
                    sd["sd_encoder"][clean_key] = v
                elif "default_decoder" in k:
                    clean_key = k.replace(".default_decoder.weight", ".weight")
                    sd["sd_decoder"][clean_key] = v
                elif "default_others" in k:
                    clean_key = k.replace(".default_others.weight", ".weight")
                    sd["sd_other"][clean_key] = v
                elif "conv_in" in k:
                    # conv_in goes to encoder
                    sd["sd_encoder"][k] = v
        
        # Save VAE weights
        # Store as if in wrapper format for compatibility with cyclegan_turbo.py
        vae_state = self.vae.state_dict()
        
        sd["sd_vae_enc"] = {}
        sd["sd_vae_dec"] = {}
        
        for k, v in vae_state.items():
            if "lora" in k or "skip" in k:
                # Add "vae." prefix to match wrapped format
                prefixed_key = f"vae.{k}"
                # Encoder-related weights
                if "encoder" in k or "skip_conv" in k:
                    sd["sd_vae_enc"][prefixed_key] = v
                # Decoder-related weights
                if "decoder" in k or "skip_conv" in k:
                    sd["sd_vae_dec"][prefixed_key] = v
        
        # Save to file
        torch.save(sd, outf)
        print(f"✓ Checkpoint saved successfully!")
        print(f"  - Encoder weights: {len(sd['sd_encoder'])} parameters")
        print(f"  - Decoder weights: {len(sd['sd_decoder'])} parameters")
        print(f"  - Other weights: {len(sd['sd_other'])} parameters")
        print(f"  - VAE encoder: {len(sd['sd_vae_enc'])} parameters")
        print(f"  - VAE decoder: {len(sd['sd_vae_dec'])} parameters")
