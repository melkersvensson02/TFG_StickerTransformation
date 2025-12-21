import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from pix2pix_turbo_modified import Pix2Pix_Turbo
from image_prep import canny_from_pil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, help='path to the input image')
    parser.add_argument('--prompt', type=str, required=True, help='the prompt to be used')
    parser.add_argument('--model_name', type=str, default='', help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default='', help='path to a model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--low_threshold', type=int, default=100, help='Canny low threshold')
    parser.add_argument('--high_threshold', type=int, default=200, help='Canny high threshold')
    parser.add_argument('--gamma', type=float, default=0.4, help='The sketch interpolation guidance amount')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument('--deterministic', type=bool, default=False, help='To control if deterministic or not')
    parser.add_argument('--use_fp16', action='store_true', help='Use Float16 precision for faster inference')
    parser.add_argument('--nakedModel', type=bool, default=False, help='if you want just the original sd-turbo')
    parser.add_argument('--nakedName', type=str, default='', help='to save the output with a different name')
    parser.add_argument("--load_unet_default", type=bool, default=False)
    args = parser.parse_args()

    # only one of model_name and model_path should be provided
    if args.model_name == '' != args.model_path == '':
        raise ValueError('Either model_name or model_path should be provided')

    os.makedirs(args.output_dir, exist_ok=True)

    # initialize the model
    model = Pix2Pix_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path, use_pipeline_loader=args.load_unet_default)
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
            # determines the exact noise latent sampled by torch.randn
            torch.manual_seed(args.seed)
            B, C, H, W = c_t.shape
            noise = torch.randn((1, 4, H // 8, W // 8), device=c_t.device)
            if args.use_fp16:
                c_t = c_t.half()
                noise = noise.half()
            output_image = model(c_t, args.prompt, deterministic=False, r=args.gamma, noise_map=noise)

        else:
            # Alwyays resize to 512x512 the input image 
            input_image = input_image.resize((512, 512), Image.LANCZOS)
            c_t = F.to_tensor(input_image).unsqueeze(0).cuda()
            if args.use_fp16:
                c_t = c_t.half()
            if args.load_unet_default:
                gen = torch.Generator(device=c_t.device).manual_seed(args.seed)
                noise = torch.randn((1, 4, 512//8, 512//8), device=c_t.device, generator=gen, dtype=c_t.dtype)
            else:
                print("Using manual seed for noise generation")
                torch.manual_seed(args.seed)
                B, C, H, W = c_t.shape
                noise = torch.randn((1, 4, H // 8, W // 8), device=c_t.device)
            output_image = model(c_t, args.prompt, deterministic=args.deterministic, r=args.gamma, noise_map=noise)
            
        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)

    # save the output image
    print("DONE!!")
    if not(args.nakedModel):
        bname = bname.replace('.png', '_output.png')
        bname = f"gammaValue_{args.gamma}_{bname}"
    else:
        bname= f"{args.nakedName}_output.png"
    print(f"\n{bname}")
    save_path = os.path.join(args.output_dir, bname)
    output_pil.save(save_path)
    print("\nSaved at:", save_path)
    """
    Prompt example:
    python src/Inference_paired.py --input_image /data/upftfg19/mfsvensson/Data_TFG/myPairedDatasets/train_A/50.png \
    --prompt "Minimal black tattoo line art, bold clean contour lines, simplified details, sticker-style, high contrast." \
    --model_path /home/mfsvensson/TFG_reps/Old-img2img-turbo/outputs/myPairedDataset_ML2_G_B/checkpoints/model_model_Fill50k_1001.pkl \
    --output_dir /home/mfsvensson/TFG_reps/Old-img2img-turbo/outputs/gammaTest/ \
    --gamma 1.0 \


    Naked version:
    python src/inference_paired.py --input_image /data/upftfg19/mfsvensson/Data_TFG/myPairedDatasets/train_A/50.png --prompt "A close-up shot of a skateboard on a colorful graffiti-filled backdrop in an urban setting, capturing the essence of street culture." --model_path /home/mfsvensson/TFG_reps/Old-img2img-turbo/outputs/myPairedDataset_ML2_G_B/checkpoints/model_model_Fill50k_1001.pkl --output_dir /home/mfsvensson/TFG_reps/Old-img2img-turbo/outputs/gammaTest/ --gamma 0.0 --nakedModel True --nakedName 'skateboardUrban'

    Version with default unet loader:
    python src/inference_paired.py --input_image /data/upftfg19/mfsvensson/Data_TFG/myPairedDatasets/train_A/50.png --prompt "a blue dog" --model_path /home/mfsvensson/TFG_reps/Old-img2img-turbo/outputs/myPairedDataset_ML2_G_B/checkpoints/model_model_Fill50k_1001.pkl --output_dir /home/mfsvensson/TFG_reps/Old-img2img-turbo/ --gamma 0.0 --nakedModel True --nakedName 'blue_dog_v2_modified.png' --load_unet_default True
    """