"""
compare_models.py — Side-by-side comparison of CycleGAN vs fine-tuned pix2pix model.

For each image it produces a 2×3 grid:
  Row 1: [Reference (masked) | CycleGAN (masked)    | Fine-tuned (masked)  ]
  Row 2: [Reference (orig)   | Fine-tuned (original) | (blank)              ]

Usage:
python src/compare_models.py --data_dir /data/upftfg19/mfsvensson/Data_TFG/dataToCompare --cyclegan_model_path /home/mfsvensson/TFG_reps/Old-img2img-turbo/outputs/many_runs_bs_3/checkpoints/model_6001.pkl --cyclegan_prompt "Minimal black tattoo line art, bold clean contour lines, simplified details, sticker-style, high contrast." --finetuned_model_path /home/mfsvensson/TFG_reps/Old-img2img-turbo/outputs/fineTune_cyclegan_a2b_semantic/checkpoints/model_step_5001.pkl --output_dir ./ComparedModels"""

import os
import sys
import json
import argparse
from PIL import Image, ImageDraw, ImageFont

import torch
from torchvision import transforms
import torchvision.transforms.functional as F

p = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, p)

from cyclegan_turbo import CycleGAN_Turbo
from pix2pix_turbo_from_cyclegan import Pix2Pix_Turbo_from_CycleGAN


# ── helpers ─────────────────────────────────────────────────────────────────

def load_and_resize(path: str, size: int = 512) -> Image.Image:
    img = Image.open(path).convert("RGB")
    T = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(size),
    ])
    return T(img)


def pil_to_cyclegan_tensor(img: Image.Image) -> torch.Tensor:
    """[0,1] → [-1,1], add batch dim, send to cuda."""
    x = transforms.ToTensor()(img)
    x = transforms.Normalize([0.5], [0.5])(x)
    return x.unsqueeze(0).cuda()


def pil_to_pix2pix_tensor(img: Image.Image) -> torch.Tensor:
    """[0,1], add batch dim, send to cuda (no normalization — matches training)."""
    x = F.to_tensor(img)
    return x.unsqueeze(0).cuda()


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Model output [-1,1] → PIL."""
    return transforms.ToPILImage()((t[0].cpu() * 0.5 + 0.5).clamp(0, 1))


def add_label(img: Image.Image, text: str, font_size: int = 18) -> Image.Image:
    """Add a white label bar at the top of the image."""
    bar_h = font_size + 8
    out = Image.new("RGB", (img.width, img.height + bar_h), (30, 30, 30))
    out.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    # centre the text horizontally
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((img.width - tw) // 2, 4), text, fill=(255, 255, 255), font=font)
    return out


def make_grid(rows: list[list[Image.Image]], gap: int = 4) -> Image.Image:
    """Paste panels in a 2-D grid (list of rows) with a thin gap."""
    col_widths = [max(row[c].width for row in rows if c < len(row)) for c in range(max(len(r) for r in rows))]
    row_heights = [max(p.height for p in row) for row in rows]
    total_w = sum(col_widths) + gap * (len(col_widths) - 1)
    total_h = sum(row_heights) + gap * (len(row_heights) - 1)
    grid = Image.new("RGB", (total_w, total_h), (60, 60, 60))
    y = 0
    for r, row in enumerate(rows):
        x = 0
        for c, panel in enumerate(row):
            grid.paste(panel, (x, y))
            x += col_widths[c] + gap
        y += row_heights[r] + gap
    return grid


# ── inference helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def run_cyclegan(model: CycleGAN_Turbo, img: Image.Image,
                 prompt: str, direction: str) -> Image.Image:
    x_t = pil_to_cyclegan_tensor(img)
    out = model(x_t, direction=direction, caption=prompt)
    return tensor_to_pil(out)


@torch.no_grad()
def run_finetuned(model: Pix2Pix_Turbo_from_CycleGAN, img: Image.Image,
                  prompt: str) -> Image.Image:
    c_t = pil_to_pix2pix_tensor(img)
    out = model(c_t, prompt=prompt)
    return tensor_to_pil(out)


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Compare CycleGAN vs fine-tuned pix2pix models.")
    parser.add_argument("--data_dir", required=True,
                        help="Folder containing test_A/, test_A_Masked/, test_prompts.json")
    parser.add_argument("--cyclegan_model_path", required=True,
                        help="Path to CycleGAN .pkl checkpoint")
    parser.add_argument("--cyclegan_prompt", default=None,
                        help="Fixed prompt for CycleGAN a2b (falls back to fixed_prompt_b.txt in data_dir)")
    parser.add_argument("--cyclegan_direction", default="a2b",
                        help="Translation direction for CycleGAN (default: a2b)")
    parser.add_argument("--finetuned_model_path", required=True,
                        help="Path to fine-tuned (pix2pix-from-cyclegan) .pkl checkpoint")
    parser.add_argument("--output_dir", default="./ComparedModels",
                        help="Directory to save comparison grids (default: ./ComparedModels)")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Size to resize all images to (default: 512)")
    parser.add_argument("--use_fp16", action="store_true",
                        help="Use FP16 for faster inference")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── directories ──────────────────────────────────────────────────────────
    dir_orig   = os.path.join(args.data_dir, "test_A")
    dir_masked = os.path.join(args.data_dir, "test_A_Masked")
    prompts_file = os.path.join(args.data_dir, "test_prompts.json")

    for d in (dir_orig, dir_masked):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Expected directory not found: {d}")
    if not os.path.isfile(prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    with open(prompts_file) as f:
        prompts: dict = json.load(f)

    # resolve cyclegan prompt: CLI arg > fixed_prompt_b.txt (b is the target domain for a2b)
    cyclegan_prompt = args.cyclegan_prompt
    if cyclegan_prompt is None:
        prompt_b_file = os.path.join(args.data_dir, "fixed_prompt_b.txt")
        if os.path.isfile(prompt_b_file):
            with open(prompt_b_file) as f:
                cyclegan_prompt = f.read().strip()
            print(f"Loaded cyclegan prompt from fixed_prompt_b.txt: '{cyclegan_prompt}'")
        else:
            raise ValueError("--cyclegan_prompt not provided and fixed_prompt_b.txt not found in data_dir")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── load models ──────────────────────────────────────────────────────────
    cyclegan = CycleGAN_Turbo(pretrained_path=args.cyclegan_model_path)
    cyclegan.eval()
    cyclegan.unet.enable_xformers_memory_efficient_attention()
    if args.use_fp16:
        cyclegan.half()

    finetuned = Pix2Pix_Turbo_from_CycleGAN(
        cyclegan_checkpoint_path=args.finetuned_model_path,
    )
    finetuned.set_eval()
    finetuned.unet.enable_xformers_memory_efficient_attention()
    if args.use_fp16:
        finetuned.half()

    # ── collect images ───────────────────────────────────────────────────────
    supported = (".png", ".jpg", ".jpeg", ".webp")
    image_names = sorted(
        f for f in os.listdir(dir_orig)
        if f.lower().endswith(supported)
    )

    if not image_names:
        raise RuntimeError(f"No images found in {dir_orig}")

    print(f"Found {len(image_names)} images — running inference...")

    for fname in image_names:
        stem = os.path.splitext(fname)[0]

        path_orig   = os.path.join(dir_orig,   fname)
        path_masked = os.path.join(dir_masked, fname)

        if not os.path.isfile(path_masked):
            print(f"  [SKIP] Masked image not found for {fname}")
            continue

        prompt = prompts.get(fname) or prompts.get(stem)
        if prompt is None:
            print(f"  [SKIP] No prompt found for {fname} in test_prompts.json")
            continue

        print(f"  Processing: {fname}")

        img_orig   = load_and_resize(path_orig,   args.image_size)
        img_masked = load_and_resize(path_masked, args.image_size)

        # 1. CycleGAN on masked (no per-image prompt)
        out_cycle = run_cyclegan(cyclegan, img_masked,
                                 cyclegan_prompt, args.cyclegan_direction)

        # 2. Fine-tuned on masked
        out_ft_masked = run_finetuned(finetuned, img_masked, prompt)

        # 3. Fine-tuned on original (unmasked)
        out_ft_orig = run_finetuned(finetuned, img_orig, prompt)

        # ── build 2×3 comparison grid ─────────────────────────────────────
        #   Row 1: Reference (masked) | CycleGAN (masked) | Fine-tuned (masked)
        #   Row 2: Reference (orig)   | Fine-tuned (orig)  | [blank]
        white = Image.new("RGB", (args.image_size, args.image_size), (255, 255, 255))
        row1 = [
            add_label(img_masked,    "Reference (masked)"),
            add_label(out_cycle,     "CycleGAN (masked)"),
            add_label(out_ft_masked, "Fine-tuned (masked)"),
        ]
        row2 = [
            add_label(img_orig,    "Reference (original)"),
            add_label(out_ft_orig, "Fine-tuned (original)"),
            add_label(white,       ""),
        ]
        grid = make_grid([row1, row2])

        out_path = os.path.join(args.output_dir, f"{stem}_comparison.png")
        grid.save(out_path)
        print(f"    Saved → {out_path}")

    print(f"\nDone. All comparisons saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
