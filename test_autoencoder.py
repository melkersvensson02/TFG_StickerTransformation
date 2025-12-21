#!/usr/bin/env python3
"""
Minimal "manual pipeline" for 1-step text-to-image that matches what the Diffusers
AutoPipelineForText2Image would do at a high level:

- load pipeline
- get tokenizer/text_encoder/unet/vae/scheduler
- set timesteps to 1
- sample latents with a seeded generator
- scale by init_noise_sigma
- scale_model_input
- unet forward
- scheduler.step using *the same latents variable*
- vae decode
- save image

Usage:
  python manual_1step_t2i.py \
    --model /data/upftfg19/mfsvensson/TFG_weights/img2img-turbo \
    --prompt "a blue dog" \
    --seed 0 \
    --height 512 --width 512 \
    --out manual_blue_dog.png

Note:
- guidance_scale is effectively 0.0 here (no classifier-free guidance).
- This is meant for debugging parity, not speed.
"""

import argparse
import math
import torch
from diffusers import AutoPipelineForText2Image


def encode_prompt(tokenizer, text_encoder, prompt: str, device: str, dtype: torch.dtype):
    """Encode a single prompt to encoder_hidden_states (no CFG)."""
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    with torch.no_grad():
        enc = text_encoder(input_ids)[0]
    return enc.to(dtype=dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="a blue dog")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32"])
    parser.add_argument("--out", type=str, default="manual_1step.png")
    parser.add_argument("--local_files_only", action="store_true", default=False)
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    # 0) Load the same pipeline as your one-liner
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)

    # Extract components
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet
    vae = pipe.vae
    sched = pipe.scheduler

    # Ensure everything is eval/no-grad
    unet.eval()
    vae.eval()
    text_encoder.eval()

    # 1) Scheduler timesteps (1-step)
    sched.set_timesteps(1, device=device)
    t = sched.timesteps[0]

    # 2) Prompt encodings
    prompt_embeds = encode_prompt(tokenizer, text_encoder, args.prompt, device, dtype=unet.dtype)

    # 3) Latents init (seeded, same shape rule as pipeline) we need to make sure the latents are the same shape as what the pipeline would create
    if args.height % 8 != 0 or args.width % 8 != 0:
        args.height = (args.height // 8) * 8
        args.width = (args.width // 8) * 8
        print(f"Adjusted height and width to be multiples of 8: {args.height} x {args.width}")

    latent_h = args.height // 8
    latent_w = args.width // 8

    gen = torch.Generator(device=device).manual_seed(args.seed)

    # Latents are usually float32 in many pipelines even when weights are fp16,
    # but for parity with your one-liner and simplicity we use unet.dtype.
    latents = torch.randn((1, 4, latent_h, latent_w), device=device, generator=gen, dtype=unet.dtype)

    # 4) Scale ONCE by init_noise_sigma (this is critical!)
    # For EulerDiscreteScheduler, init_noise_sigma is typically != 1.
    latents = latents * sched.init_noise_sigma

    # 5) Scale model input (scheduler-specific)
    latent_model_input = sched.scale_model_input(latents, t)

    # 6) UNet forward
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

    # 7) Denoising step: IMPORTANT — pass the SAME `latents` you scaled/updated
    latents = sched.step(noise_pred, t, latents, return_dict=True).prev_sample

    # 8) Decode
    with torch.no_grad():
        image = vae.decode(latents / vae.config.scaling_factor).sample

    # Convert to [0, 1] and save
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu()

    # Use torchvision if available, else fallback to PIL
    try:
        import torchvision
        torchvision.utils.save_image(image, args.out)
    except Exception:
        from PIL import Image
        # image: (1,3,H,W)
        img = (image[0].permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        Image.fromarray(img).save(args.out)

    print(f"Saved: {args.out}")
    print(f"Scheduler: {sched.__class__.__name__}")
    print(f"timestep: {int(t)}")
    print(f"init_noise_sigma: {float(getattr(sched, 'init_noise_sigma', math.nan))}")


if __name__ == "__main__":
    main()
