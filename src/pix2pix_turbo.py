import os
import requests
import sys
import copy
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, AutoPipelineForText2Image
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig
p = "src/"
sys.path.append(p)
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd


class TwinConv(torch.nn.Module):
    # TwinConv stores two conv layers: a frozen copy of the pretrained one, and a trainable copy (“curr”)
    def __init__(self, convin_pretrained, convin_curr):
        super(TwinConv, self).__init__()
        self.conv_in_pretrained = copy.deepcopy(convin_pretrained)
        self.conv_in_curr = copy.deepcopy(convin_curr)
        self.r = None

    def forward(self, x):
        # gamma will condition the interpolation between the two conv layers
        # if r = 1 -> use only the pretrained conv
        x1 = self.conv_in_pretrained(x).detach()
        x2 = self.conv_in_curr(x)
        return x1 * (1 - self.r) + x2 * (self.r)


class Pix2Pix_Turbo(torch.nn.Module):
    def __init__(self, pretrained_name=None, pretrained_path=None, ignore_skip=False, skip_weight=1.0,ckpt_folder="checkpoints", lora_rank_unet=8, lora_rank_vae=4
                 ,use_pipeline_loader=False):
        super().__init__()
        if use_pipeline_loader:
            pipe = AutoPipelineForText2Image.from_pretrained(
                "/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo",
                torch_dtype=torch.float32,
                local_files_only=True,
            ).to("cuda")

            # Optional: expose internals so the rest of your code can reuse them
            #self.pipe = pipe
            self.tokenizer = pipe.tokenizer
            self.text_encoder = pipe.text_encoder
            unet = pipe.unet
            vae = pipe.vae
            self.sched = pipe.scheduler
            self.sched.set_timesteps(1, device="cuda")
            self.timesteps = self.sched.timesteps[0]
            print("Loaded UNet and VAE from pipeline loader")   
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo", subfolder="tokenizer", local_files_only=True)
            self.text_encoder = CLIPTextModel.from_pretrained("/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo", subfolder="text_encoder").cuda()
            self.sched = make_1step_sched()
            vae = AutoencoderKL.from_pretrained("/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo", subfolder="vae")
            unet = UNet2DConditionModel.from_pretrained("/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo", subfolder="unet")

        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = ignore_skip
        vae.decoder.skip_weight = skip_weight

        if pretrained_name == "edge_to_image":
            #url = "/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo/edge_to_image_loras.pkl"
            url = "./edge_to_image_loras.pkl"
            os.makedirs(ckpt_folder, exist_ok=True)
            outf = os.path.join(ckpt_folder, "edge_to_image_loras.pkl")
            if not os.path.exists(outf):
                print(f"Downloading checkpoint to {outf}")
                response = requests.get(url, stream=True)
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(outf, 'wb') as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
                print(f"Downloaded successfully to {outf}")
            p_ckpt = outf
            sd = torch.load(p_ckpt, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            # Load LoRA weights into VAE
            # Base model loads + LoRA weights are injected and added to forward output.¨
            # We are at inference time, so we alredy have trained LoRA weights to load.
            print("WE ARE HERE!")
            """
            _sd_vae = vae.state_dict()
            for key in _sd_vae:
                if "skip_conv" in key:
                    print(key)
            for k in sd["state_dict_vae"]:
                print(f"{k}")
                k_mapped = k
                #k_mapped = k.replace(".base_layer", "")
                _sd_vae[k_mapped] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)

            """
            print("WE ARE HERE!")

            _sd_vae = vae.state_dict()

            # Check what's in the checkpoint BEFORE loading
            print("\n=== Checkpoint keys with skip_conv ===")
            for k in sd["state_dict_vae"]:
                if "skip_conv" in k:
                    print(f"Checkpoint key: {k}")
                    print(f"  Shape: {sd['state_dict_vae'][k].shape}")
                    print(f"  Mean: {sd['state_dict_vae'][k].mean():.6f}, Std: {sd['state_dict_vae'][k].std():.6f}")

            # Check what's in _sd_vae BEFORE loading
            print("\n=== Model keys with skip_conv ===")
            for key in _sd_vae:
                if "skip_conv" in key:
                    print(f"Model key: {key}")
                    print(f"  Shape: {_sd_vae[key].shape}")
                    print(f"  Mean: {_sd_vae[key].mean():.6f}, Std: {_sd_vae[key].std():.6f}")

            # Now do the loading
            for k in sd["state_dict_vae"]:
                k_mapped = k
                _sd_vae[k_mapped] = sd["state_dict_vae"][k]

            # Check AFTER assignment to _sd_vae
            print("\n=== After assignment to _sd_vae ===")
            for key in _sd_vae:
                if "skip_conv_1.base_layer" in key:
                    print(f"Key: {key}")
                    print(f"  Mean: {_sd_vae[key].mean():.6f}, Std: {_sd_vae[key].std():.6f}")

            result = vae.load_state_dict(_sd_vae, strict=False)
            print("\n=== load_state_dict result ===")
            print(f"Missing keys: {len(result.missing_keys)}")
            print(f"Unexpected keys: {len(result.unexpected_keys)}")
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_name == "sketch_to_image_stochastic":
            # download from url
            url = "/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo/sketch_to_image_stochastic_lora.pkl"
            os.makedirs(ckpt_folder, exist_ok=True)
            outf = os.path.join(ckpt_folder, "sketch_to_image_stochastic_lora.pkl")
            if not os.path.exists(outf):
                print(f"Downloading checkpoint to {outf}")
                response = requests.get(url, stream=True)
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(outf, 'wb') as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
                print(f"Downloaded successfully to {outf}")
            p_ckpt = outf
            convin_pretrained = copy.deepcopy(unet.conv_in)
            # replace unet.conv_in with TwinConv
            unet.conv_in = TwinConv(convin_pretrained, unet.conv_in)
            sd = torch.load(p_ckpt, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae, strict=False)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet, strict=False)

        elif pretrained_name is None and pretrained_path is None:
            print("Initializing model with random weights")
            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]
            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            target_modules_unet = [
                "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
            ]
            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
                target_modules=target_modules_unet
            )
            unet.add_adapter(unet_lora_config)
            self.lora_rank_unet = lora_rank_unet
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae
            self.target_modules_unet = target_modules_unet

        # unet.enable_xformers_memory_efficient_attention()
        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([999], device="cuda").long()
        self.text_encoder.requires_grad_(False)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

    def forward(self, c_t, prompt=None, prompt_tokens=None, deterministic=True, r=1.0, noise_map=None):
        # either the prompt or the prompt_tokens should be provided
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"

        if prompt is not None:
            # encode the text prompt
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]
        if deterministic:
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc,).sample
            x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
            x_denoised = x_denoised.to(model_pred.dtype)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        else:
            # scale the lora weights based on the r value
            self.unet.set_adapters(["default"], weights=[r])
            set_weights_and_activate_adapters(self.vae, ["vae_skip"], [r])
            c_t = c_t.to(device=self.vae.device, dtype=self.vae.dtype)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            # combine the input and noise
            unet_input = encoded_control * r + noise_map * (1 - r)
            # very imp to decide r here too for the TwinConv
            self.unet.conv_in.r = r
            # forward pass (Runs the UNet forward on unet_input at the single timestep)
            unet_input = unet_input.to(
                device=self.unet.device,
                dtype=self.unet.dtype,
            )
            if r == 0.0:
                t = self.timesteps
                unet_output = self.unet(unet_input, t, self.timesteps, encoder_hidden_states=caption_enc,).sample
            else:
                unet_output = self.unet(unet_input, self.timesteps, encoder_hidden_states=caption_enc,).sample
            self.unet.conv_in.r = None
            # ?? denoising step
            if r == 0.0:
                x_denoised = self.sched.step(unet_output, t, unet_input).prev_sample
            else:
                x_denoised = self.sched.step(unet_output, self.timesteps, unet_input, return_dict=True).prev_sample
            x_denoised = x_denoised.to(unet_output.dtype)
            # maakes sure to use the correct skip activations
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            self.vae.decoder.gamma = r
            # runs the decoder forward
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        return output_image

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        torch.save(sd, outf)
