# Evaluates a trained CycleGAN_Turbo model on paired test images (A and B domains).
# Generates side-by-side comparisons of input and translated outputs for visual inspection.
# Supports random or sequential sampling with configurable image count.

import os
import argparse
import random
from glob import glob
from PIL import Image
import torch
from torchvision import transforms
from cyclegan_turbo import CycleGAN_Turbo
from my_utils.training_utils import build_transform
from model import make_1step_sched
from tqdm import tqdm


def get_paired_images(dataset_folder, max_images=50):
    """
    Get paired images from train_A and train_B folders.
    Returns list of tuples (path_a, path_b) with matching filenames.
    """
    # Get all images from train_A
    l_images_a = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_a.extend(glob(os.path.join(dataset_folder, "train_A", ext)))
    
    # Get all images from train_B
    l_images_b = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_b.extend(glob(os.path.join(dataset_folder, "train_B", ext)))
    
    # Create dictionaries with basename as key
    dict_a = {os.path.basename(p): p for p in l_images_a}
    dict_b = {os.path.basename(p): p for p in l_images_b}
    
    # Find matching pairs
    paired_images = []
    for name in dict_a.keys():
        if name in dict_b:
            paired_images.append((dict_a[name], dict_b[name]))
    
    print(f"Found {len(paired_images)} paired images")
    
    # Randomly sample up to max_images
    if len(paired_images) > max_images:
        paired_images = random.sample(paired_images, max_images)
        print(f"Randomly selected {max_images} paired images")
    
    return paired_images


def load_fixed_prompts(dataset_folder):
    """
    Load fixed prompts from fixed_prompt_A.txt and fixed_prompt_B.txt
    """
    prompt_a_path = os.path.join(dataset_folder, "fixed_prompt_a.txt")
    prompt_b_path = os.path.join(dataset_folder, "fixed_prompt_b.txt")
    
    with open(prompt_a_path, 'r') as f:
        prompt_a = f.read().strip()
    
    with open(prompt_b_path, 'r') as f:
        prompt_b = f.read().strip()
    
    return prompt_a, prompt_b


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, required=True, 
                        help='path to dataset folder containing train_A, train_B, test_A, test_B, fixed_prompt_A.txt, fixed_prompt_B.txt')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='path to the trained model checkpoint (.pkl file)')
    parser.add_argument('--output_dir', type=str, default='cycle_test_output', 
                        help='directory to save cycle test results')
    parser.add_argument('--max_images', type=int, default=50, 
                        help='maximum number of paired images to test (default: 50)')
    parser.add_argument('--image_prep', type=str, default='resize_512x512', 
                        help='image preparation method (default: resize_512x512)')
    parser.add_argument('--use_fp16', action='store_true', 
                        help='use Float16 precision for faster inference')
    parser.add_argument('--seed', type=int, default=42, 
                        help='random seed for reproducibility (default: 42)')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    cycle_a2b2a_dir = os.path.join(args.output_dir, "cycle_A2B2A")
    cycle_b2a2b_dir = os.path.join(args.output_dir, "cycle_B2A2B")
    os.makedirs(cycle_a2b2a_dir, exist_ok=True)
    os.makedirs(cycle_b2a2b_dir, exist_ok=True)
    
    print(f"Loading model from {args.model_path}")
    # Initialize the model
    model = CycleGAN_Turbo(pretrained_path=args.model_path)
    model.eval()
    model.unet.enable_xformers_memory_efficient_attention()
    if args.use_fp16:
        model.half()
    
    # Get the networks from the model
    vae_enc = model.vae_enc
    vae_dec = model.vae_dec
    unet = model.unet
    noise_scheduler_1step = make_1step_sched()
    
    # Load fixed prompts
    print(f"Loading prompts from {args.dataset_folder}")
    prompt_a, prompt_b = load_fixed_prompts(args.dataset_folder)
    print(f"Prompt A: {prompt_a}")
    print(f"Prompt B: {prompt_b}")
    
    # Encode prompts
    tokenizer = model.tokenizer
    text_encoder = model.text_encoder
    
    tokens_a = tokenizer(prompt_a, max_length=tokenizer.model_max_length, 
                         padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
    tokens_b = tokenizer(prompt_b, max_length=tokenizer.model_max_length, 
                         padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
    
    with torch.no_grad():
        fixed_a2b_emb = text_encoder(tokens_b)[0].detach()
        fixed_b2a_emb = text_encoder(tokens_a)[0].detach()
    
    if args.use_fp16:
        fixed_a2b_emb = fixed_a2b_emb.half()
        fixed_b2a_emb = fixed_b2a_emb.half()
    
    # Build transform
    T_val = build_transform(args.image_prep)
    
    # Get paired images
    print(f"Getting paired images from {args.dataset_folder}")
    paired_images = get_paired_images(args.dataset_folder, args.max_images)
    
    # Prepare timesteps
    timesteps = torch.tensor([noise_scheduler_1step.config.num_train_timesteps - 1], device="cuda").long()
    
    print(f"\nStarting cycle testing on {len(paired_images)} image pairs...")
    print("=" * 80)
    
    # Process each paired image
    for idx, (path_a, path_b) in enumerate(tqdm(paired_images, desc="Processing cycles")):
        base_name = os.path.splitext(os.path.basename(path_a))[0]
        
        # Load and preprocess images
        img_a_pil = Image.open(path_a).convert('RGB')
        img_b_pil = Image.open(path_b).convert('RGB')
        
        img_a_pil_transformed = T_val(img_a_pil)
        img_b_pil_transformed = T_val(img_b_pil)
        
        # Convert to tensors
        img_a = transforms.ToTensor()(img_a_pil_transformed)
        img_a = transforms.Normalize([0.5], [0.5])(img_a).unsqueeze(0).cuda()
        
        img_b = transforms.ToTensor()(img_b_pil_transformed)
        img_b = transforms.Normalize([0.5], [0.5])(img_b).unsqueeze(0).cuda()
        
        if args.use_fp16:
            img_a = img_a.half()
            img_b = img_b.half()
        
        with torch.no_grad():
            # Cycle A -> B -> A
            cyc_fake_b = CycleGAN_Turbo.forward_with_networks(
                img_a, "a2b", vae_enc, unet, vae_dec, 
                noise_scheduler_1step, timesteps, fixed_a2b_emb
            )
            cyc_rec_a = CycleGAN_Turbo.forward_with_networks(
                cyc_fake_b, "b2a", vae_enc, unet, vae_dec, 
                noise_scheduler_1step, timesteps, fixed_b2a_emb
            )
            
            # Cycle B -> A -> B
            cyc_fake_a = CycleGAN_Turbo.forward_with_networks(
                img_b, "b2a", vae_enc, unet, vae_dec, 
                noise_scheduler_1step, timesteps, fixed_b2a_emb
            )
            cyc_rec_b = CycleGAN_Turbo.forward_with_networks(
                cyc_fake_a, "a2b", vae_enc, unet, vae_dec, 
                noise_scheduler_1step, timesteps, fixed_a2b_emb
            )
        
        # Convert tensors to PIL images
        # Helper function to denormalize and convert
        def tensor_to_pil(tensor):
            tensor = tensor[0].cpu().float() * 0.5 + 0.5  # denormalize from [-1, 1] to [0, 1]
            tensor = tensor.clamp(0, 1)
            return transforms.ToPILImage()(tensor)
        
        img_a_save = tensor_to_pil(img_a)
        cyc_fake_b_save = tensor_to_pil(cyc_fake_b)
        cyc_rec_a_save = tensor_to_pil(cyc_rec_a)
        
        img_b_save = tensor_to_pil(img_b)
        cyc_fake_a_save = tensor_to_pil(cyc_fake_a)
        cyc_rec_b_save = tensor_to_pil(cyc_rec_b)
        
        # Save A -> B -> A cycle
        img_a_save.save(os.path.join(cycle_a2b2a_dir, f"{base_name}_input_A.png"))
        cyc_fake_b_save.save(os.path.join(cycle_a2b2a_dir, f"{base_name}_intermediate_B.png"))
        cyc_rec_a_save.save(os.path.join(cycle_a2b2a_dir, f"{base_name}_reconstructed_A.png"))
        
        # Save B -> A -> B cycle
        img_b_save.save(os.path.join(cycle_b2a2b_dir, f"{base_name}_input_B.png"))
        cyc_fake_a_save.save(os.path.join(cycle_b2a2b_dir, f"{base_name}_intermediate_A.png"))
        cyc_rec_b_save.save(os.path.join(cycle_b2a2b_dir, f"{base_name}_reconstructed_B.png"))
    
    print("=" * 80)
    print(f"\nCycle testing complete!")
    print(f"Results saved to:")
    print(f"  A->B->A cycles: {cycle_a2b2a_dir}")
    print(f"  B->A->B cycles: {cycle_b2a2b_dir}")
    print(f"\nFor each image pair '{base_name}', you'll find:")
    print(f"  - {base_name}_input_A/B.png: Original input image")
    print(f"  - {base_name}_intermediate_B/A.png: Translated to other domain")
    print(f"  - {base_name}_reconstructed_A/B.png: Reconstructed back to original domain")
