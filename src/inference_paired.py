import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from pix2pix_turbo import Pix2Pix_Turbo
from image_prep import canny_from_pil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True, help='path to the input image')
    parser.add_argument('--prompt', type=str, required=True, help='the prompt to be used')
    parser.add_argument('--model_name', type=str, default='', help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default='', help='path to a model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--low_threshold', type=int, default=100, help='Canny low threshold')
    parser.add_argument('--high_threshold', type=int, default=200, help='Canny high threshold')
    parser.add_argument('--gamma', type=float, default=0.4, help='The sketch interpolation guidance amount')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument('--use_fp16', action='store_true', help='Use Float16 precision for faster inference')
    args = parser.parse_args()

    # only one of model_name and model_path should be provided
    if args.model_name == '' != args.model_path == '':
        raise ValueError('Either model_name or model_path should be provided')

    os.makedirs(args.output_dir, exist_ok=True)

    # initialize the model
    model = Pix2Pix_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    model.set_eval()
    if args.use_fp16:
        model.half()

    # make sure that the input image is a multiple of 8
    input_image = Image.open(args.input_image).convert('RGB')
    new_width = input_image.width - input_image.width % 8
    new_height = input_image.height - input_image.height % 8
    input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
    bname = os.path.basename(args.input_image)

    # translate the image
    with torch.no_grad():
        if args.model_name == 'edge_to_image':
            canny = canny_from_pil(input_image, args.low_threshold, args.high_threshold)
            canny_viz_inv = Image.fromarray(255 - np.array(canny))
            canny_viz_inv.save(os.path.join(args.output_dir, bname.replace('.png', '_canny.png')))
            c_t = F.to_tensor(canny).unsqueeze(0).cuda()
            if args.use_fp16:
                c_t = c_t.half()
            output_image = model(c_t, args.prompt)

        elif args.model_name == 'sketch_to_image_stochastic':
            image_t = F.to_tensor(input_image) < 0.5
            c_t = image_t.unsqueeze(0).cuda().float()
            torch.manual_seed(args.seed)
            B, C, H, W = c_t.shape
            noise = torch.randn((1, 4, H // 8, W // 8), device=c_t.device)
            if args.use_fp16:
                c_t = c_t.half()
                noise = noise.half()
            output_image = model(c_t, args.prompt, deterministic=False, r=args.gamma, noise_map=noise)

        else:
            c_t = F.to_tensor(input_image).unsqueeze(0).cuda()
            if args.use_fp16:
                c_t = c_t.half()
            output_image = model(c_t, args.prompt)

        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)

    # save the output image
    bname = bname.replace('.png', '_output.png')
    output_pil.save(os.path.join(args.output_dir, bname))

    # Add this at the end of inference_paired.py, after model initialization
    
    print("\n" + "="*60)
    print("WEIGHT INSPECTION")
    print("="*60)
    
    # 1. VAE Decoder Skip Convs
    print("\n--- VAE Decoder Skip Connections ---")
    for i in [1, 2, 3, 4]:
        skip_conv = getattr(model.vae.decoder, f'skip_conv_{i}')
        w = skip_conv.weight
        print(f"skip_conv_{i}: shape={list(w.shape)}, mean={w.mean().item():.6e}, "
              f"std={w.std().item():.6e}, norm={w.norm().item():.6e}")
    
    # 2. UNet conv_in (for sketch model)
    print("\n--- UNet conv_in ---")
    if hasattr(model.unet.conv_in, 'conv_in_pretrained'):
        # TwinConv case
        print("TwinConv detected:")
        w1 = model.unet.conv_in.conv_in_pretrained.weight
        w2 = model.unet.conv_in.conv_in_curr.weight
        print(f"  conv_in_pretrained: mean={w1.mean().item():.6e}, std={w1.std().item():.6e}")
        print(f"  conv_in_curr: mean={w2.mean().item():.6e}, std={w2.std().item():.6e}")
    else:
        w = model.unet.conv_in.weight
        print(f"conv_in: shape={list(w.shape)}, mean={w.mean().item():.6e}, "
              f"std={w.std().item():.6e}, norm={w.norm().item():.6e}")
    
    # 3. Sample UNet LoRA weights
    print("\n--- UNet LoRA Samples ---")
    lora_count = 0
    for name, param in model.unet.named_parameters():
        if 'lora' in name and lora_count < 3:
            print(f"{name}: shape={list(param.shape)}, mean={param.mean().item():.6e}, "
                  f"std={param.std().item():.6e}")
            lora_count += 1
    
    # 4. Sample VAE LoRA weights
    print("\n--- VAE LoRA Samples ---")
    lora_count = 0
    for name, param in model.vae.named_parameters():
        if 'lora' in name and lora_count < 3:
            print(f"{name}: shape={list(param.shape)}, mean={param.mean().item():.6e}, "
                  f"std={param.std().item():.6e}")
            lora_count += 1
    
    # 5. Sample UNet attention block
    print("\n--- UNet Down Block 0 Attentions ---")
    try:
        attn = model.unet.down_blocks[0].attentions[0]
        for name in ['to_q', 'to_k', 'to_v', 'to_out']:
            if hasattr(attn.transformer_blocks[0].attn1, name):
                layer = getattr(attn.transformer_blocks[0].attn1, name)
                if hasattr(layer, 'weight'):
                    w = layer.weight
                elif isinstance(layer, torch.nn.ModuleList) or isinstance(layer, torch.nn.Sequential):
                    w = layer[0].weight
                else:
                    continue
                print(f"  {name}: mean={w.mean().item():.6e}, std={w.std().item():.6e}")
    except Exception as e:
        print(f"  Could not inspect attention: {e}")
    
    # 6. Sample VAE encoder block
    print("\n--- VAE Encoder Down Block 0 ---")
    try:
        block = model.vae.encoder.down_blocks[0]
        if hasattr(block, 'resnets'):
            res = block.resnets[0]
            if hasattr(res, 'conv1'):
                w = res.conv1.weight
                print(f"  conv1: mean={w.mean().item():.6e}, std={w.std().item():.6e}")
            if hasattr(res, 'conv2'):
                w = res.conv2.weight
                print(f"  conv2: mean={w.mean().item():.6e}, std={w.std().item():.6e}")
    except Exception as e:
        print(f"  Could not inspect encoder: {e}")
    
    # 7. Text encoder check
    print("\n--- Text Encoder Sample ---")
    w = model.text_encoder.text_model.encoder.layers[0].self_attn.q_proj.weight
    print(f"Layer 0 q_proj: mean={w.mean().item():.6e}, std={w.std().item():.6e}")
    
    # 8. Library versions
    print("\n--- Library Versions ---")
    import torch, diffusers, transformers, peft
    print(f"torch: {torch.__version__}")
    print(f"diffusers: {diffusers.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"peft: {peft.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 9. Model dtype and device info
    print("\n--- Model State ---")
    print(f"UNet dtype: {next(model.unet.parameters()).dtype}")
    print(f"VAE dtype: {next(model.vae.parameters()).dtype}")
    print(f"Text encoder dtype: {next(model.text_encoder.parameters()).dtype}")
    print(f"FP16 mode: {args.use_fp16}")
    
    # 10. LoRA adapter info
    print("\n--- LoRA Adapter Info ---")
    if hasattr(model.unet, 'get_active_adapters'):
        print(f"UNet active adapters: {model.unet.get_active_adapters()}")
    if hasattr(model.vae, 'get_active_adapters'):
        print(f"VAE active adapters: {model.vae.get_active_adapters()}")
    
    # 11. Checkpoint file hash (to verify download)
    print("\n--- Checkpoint File Info ---")
    import hashlib
    if args.model_name == 'edge_to_image':
        ckpt_path = 'checkpoints/edge_to_image_loras.pkl'
    elif args.model_name == 'sketch_to_image_stochastic':
        ckpt_path = 'checkpoints/sketch_to_image_stochastic_lora.pkl'
    else:
        ckpt_path = args.model_path
    
    if ckpt_path and os.path.exists(ckpt_path):
        import hashlib
        with open(ckpt_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        file_size = os.path.getsize(ckpt_path)
        print(f"Checkpoint: {ckpt_path}")
        print(f"Size: {file_size / 1024 / 1024:.2f} MB")
        print(f"MD5: {file_hash}")
    
    # 12. Sample latent encoding (to check VAE encoder)
    print("\n--- Sample VAE Encoding ---")
    test_tensor = torch.randn(1, 3, 512, 512).cuda()
    if args.use_fp16:
        test_tensor = test_tensor.half()
    with torch.no_grad():
        encoded = model.vae.encode(test_tensor).latent_dist.sample()
    print(f"Input: mean={test_tensor.mean().item():.6e}, std={test_tensor.std().item():.6e}")
    print(f"Encoded: mean={encoded.mean().item():.6e}, std={encoded.std().item():.6e}")
    
    print("\n" + "="*60)
    print("END WEIGHT INSPECTION")
    print("="*60 + "\n")


