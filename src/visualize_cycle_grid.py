# Generates a 2x3 cycle-consistency visualization grid for a trained CycleGAN_Turbo model.
# Each row shows the original image, its translation, and the reconstructed cycle-back image.
# Useful for qualitatively evaluating cycle-consistency and translation quality.
"""
Generates a 2x3 cycle-consistency grid for each paired image in the test set.

Grid layout:
  Row 1: A  |  B_hat (A→B)      |  A_hat_rec (B_hat→A)
  Row 2: B  |  A_hat (B→A)      |  B_hat_rec (A_hat→B)

Usage:
  python src/visualize_cycle_grid.py --model_path /home/mfsvensson/TFG_reps/Old-img2img-turbo/outputs/many_runs_bs_3/checkpoints/model_4001.pkl --dataset_folder /data/upftfg19/mfsvensson/Data_TFG/dataToCompare --output_dir ./cycleObservation_MR3_4001 


  Alternatively, if fixed_prompt_a.txt / fixed_prompt_b.txt exist in dataset_folder,
  --prompt_a / --prompt_b can be omitted.
"""

import os
import sys
import argparse
from glob import glob

import torch
from PIL import Image
from torchvision import transforms
import torchvision.utils as vutils

sys.path.append("src/")
from cyclegan_turbo import CycleGAN_Turbo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_transform(resolution=512):
    return transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])


def load_image(path, transform):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)  # (1, 3, H, W)


def tensor_to_pil(t):
    """Convert a single (1, 3, H, W) or (3, H, W) tensor in [-1,1] to a PIL image."""
    t = t.squeeze(0).float().cpu()
    t = (t * 0.5 + 0.5).clamp(0, 1)
    return transforms.ToPILImage()(t)


def make_grid_image(imgs, nrow=3):
    """imgs: list of (1,3,H,W) tensors in [-1,1]. Returns a PIL image."""
    grid_t = torch.cat(imgs, dim=0)              # (N, 3, H, W)
    grid = vutils.make_grid(
        (grid_t.float() * 0.5 + 0.5).clamp(0, 1),
        nrow=nrow,
        padding=4,
        pad_value=1.0,
    )
    return transforms.ToPILImage()(grid)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def translate(model, x, direction, caption):
    """Run a single forward pass (direction: 'a2b' or 'b2a')."""
    return model(x, direction=direction, caption=caption)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # --- resolve prompts ---
    prompt_a = args.prompt_a
    prompt_b = args.prompt_b

    prompt_a_file = os.path.join(args.dataset_folder, "fixed_prompt_a.txt")
    prompt_b_file = os.path.join(args.dataset_folder, "fixed_prompt_b.txt")

    if prompt_a is None:
        if os.path.isfile(prompt_a_file):
            with open(prompt_a_file) as f:
                prompt_a = f.read().strip()
            print(f"Loaded prompt_a from file: '{prompt_a}'")
        else:
            raise ValueError("--prompt_a not provided and fixed_prompt_a.txt not found in dataset_folder")

    if prompt_b is None:
        if os.path.isfile(prompt_b_file):
            with open(prompt_b_file) as f:
                prompt_b = f.read().strip()
            print(f"Loaded prompt_b from file: '{prompt_b}'")
        else:
            raise ValueError("--prompt_b not provided and fixed_prompt_b.txt not found in dataset_folder")

    # --- load model ---
    print(f"Loading model from {args.model_path} ...")
    model = CycleGAN_Turbo(pretrained_path=args.model_path)
    model.eval()
    model.requires_grad_(False)
    print("Model loaded.")

    # --- collect paired image paths ---
    src_dir = os.path.join(args.dataset_folder, "test_A_Masked")
    tgt_dir = os.path.join(args.dataset_folder, "test_B")

    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    src_paths = sorted(p for ext in exts for p in glob(os.path.join(src_dir, ext)))

    if len(src_paths) == 0:
        raise FileNotFoundError(f"No images found in {src_dir}")

    T = build_transform(args.resolution)

    for src_path in src_paths:
        name = os.path.basename(src_path)
        tgt_path = None
        # match by filename (same name, any extension)
        stem = os.path.splitext(name)[0]
        for ext in ["png", "jpg", "jpeg", "bmp"]:
            candidate = os.path.join(tgt_dir, f"{stem}.{ext}")
            if os.path.isfile(candidate):
                tgt_path = candidate
                break

        if tgt_path is None:
            print(f"[SKIP] No matching target for {name}")
            continue

        # --- load images ---
        img_a = load_image(src_path, T).cuda()   # domain A
        img_b = load_image(tgt_path, T).cuda()   # domain B

        # --- forward passes ---
        # prompt_b is used for a2b (generating B-like output)
        # prompt_a is used for b2a (generating A-like output)
        b_hat     = translate(model, img_a, "a2b", prompt_b)   # A → B
        a_hat_rec = translate(model, b_hat, "b2a", prompt_a)   # B_hat → A  (cycle)
        a_hat     = translate(model, img_b, "b2a", prompt_a)   # B → A
        b_hat_rec = translate(model, a_hat, "a2b", prompt_b)   # A_hat → B  (cycle)

        # --- assemble 2x3 grid ---
        # Row 1: A | B_hat | A_hat_rec
        # Row 2: B | A_hat | B_hat_rec
        grid_img = make_grid_image(
            [img_a, b_hat, a_hat_rec,
             img_b, a_hat, b_hat_rec],
            nrow=3,
        )

        out_path = os.path.join(args.output_dir, f"{stem}_grid.png")
        grid_img.save(out_path)
        print(f"Saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path to the trained CycleGAN_Turbo .pkl checkpoint.")
    parser.add_argument("--dataset_folder", required=True,
                        help="Root dataset folder containing test_A_Masked/ and test_B/.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory where the grid images will be saved.")
    parser.add_argument("--prompt_a", default=None,
                        help="Text prompt describing domain A (used for b2a). "
                             "Falls back to fixed_prompt_a.txt if not provided.")
    parser.add_argument("--prompt_b", default=None,
                        help="Text prompt describing domain B (used for a2b). "
                             "Falls back to fixed_prompt_b.txt if not provided.")
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()
    main(args)
