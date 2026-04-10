# Pix2Pix_Turbo model variant that initializes from a CycleGAN_Turbo checkpoint.
# Enables paired fine-tuning on top of an already trained unpaired CycleGAN model.
# Saves checkpoints in CycleGAN-compatible format to reuse the existing inference pipeline.
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
    This class is designed to:
    - Load a trained CycleGAN model
    - Fine-tune it on paired data using pix2pix losses
    - Save checkpoints compatible with cyclegan_turbo.py for inference
    """
    # constructor method with arguments for checkpoint path and LoRA ranks
    def __init__(
        self, 
        cyclegan_checkpoint_path,
        lora_rank_unet=8, 
        lora_rank_vae=4,
        ignore_skip=False,
        skip_weight=1.0
    ):
        super().__init__()
        
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
        vae = AutoencoderKL.from_pretrained(
            "/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo", 
            subfolder="vae"
        )
        
        # Replace VAE encoder/decoder forward methods with custom versions
        # These custom versions capture skip connections during encoding
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        
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
        unet = UNet2DConditionModel.from_pretrained(
            "/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo", 
            subfolder="unet"
        )
        
        self._load_cyclegan_checkpoint(
            unet, vae, cyclegan_checkpoint_path
        )

        unet.to("cuda")
        vae.to("cuda")

        self.unet = unet
        self.vae = vae
        self.timesteps = torch.tensor([999], device="cuda").long()
    
    def _load_cyclegan_checkpoint(self, unet, vae, checkpoint_path):
        sd = torch.load(checkpoint_path, map_location="cpu")

        required_keys = ["sd_encoder", "sd_decoder", "sd_other", "sd_vae_enc", "sd_vae_dec"]
        for key in required_keys:
            if key not in sd:
                raise ValueError(f"Checkpoint missing required key: {key}")

        # Always use the ranks stored in the checkpoint to avoid shape mismatches
        rank_unet = sd["rank_unet"]
        rank_vae  = sd["rank_vae"]
        self.lora_rank_unet = rank_unet
        self.lora_rank_vae  = rank_vae

        lora_conf_encoder = LoraConfig(
            r=rank_unet, init_lora_weights="gaussian",
            target_modules=sd["l_target_modules_encoder"], lora_alpha=rank_unet
        )
        lora_conf_decoder = LoraConfig(
            r=rank_unet, init_lora_weights="gaussian",
            target_modules=sd["l_target_modules_decoder"], lora_alpha=rank_unet
        )
        lora_conf_others = LoraConfig(
            r=rank_unet, init_lora_weights="gaussian",
            target_modules=sd["l_modules_others"], lora_alpha=rank_unet
        )

        unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        unet.add_adapter(lora_conf_others, adapter_name="default_others")

        for n, p in unet.named_parameters():
            if "lora" in n and "default_encoder" in n:
                name_sd = n.replace(".default_encoder.weight", ".weight")
                if name_sd in sd["sd_encoder"]:
                    p.data.copy_(sd["sd_encoder"][name_sd])

        for n, p in unet.named_parameters():
            if "lora" in n and "default_decoder" in n:
                name_sd = n.replace(".default_decoder.weight", ".weight")
                if name_sd in sd["sd_decoder"]:
                    p.data.copy_(sd["sd_decoder"][name_sd])

        for n, p in unet.named_parameters():
            if "lora" in n and "default_others" in n:
                name_sd = n.replace(".default_others.weight", ".weight")
                if name_sd in sd["sd_other"]:
                    p.data.copy_(sd["sd_other"][name_sd])

        unet.set_adapters(["default_encoder", "default_decoder", "default_others"])

        vae_lora_config = LoraConfig(
            r=rank_vae, init_lora_weights="gaussian",
            target_modules=sd["vae_lora_target_modules"]
        )
        vae.add_adapter(vae_lora_config, adapter_name="vae_skip")

        vae_state_dict = vae.state_dict()
        for src_sd in (sd["sd_vae_enc"], sd["sd_vae_dec"]):
            for key, value in src_sd.items():
                if key.startswith("vae.") and not key.startswith("vae_b2a."):
                    clean_key = key[len("vae."):]
                    if clean_key in vae_state_dict:
                        vae_state_dict[clean_key] = value
        vae.load_state_dict(vae_state_dict, strict=False)

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
        Does a whole lot of more cheking and processing to ensure compatibility than source code, need ? not really.
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
