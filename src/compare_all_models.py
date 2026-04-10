# Compares multiple CycleGAN and pix2pix model checkpoints side-by-side on a test dataset.
# Renders a grid image grouping models by type (reference, semantic, non-semantic) for visual evaluation.
# Reads model configurations from a JSON file to flexibly compare arbitrary checkpoint collections.

import os
import sys
import json
import argparse
from PIL import Image, ImageDraw, ImageFont

import torch
from torchvision import transforms
import torchvision.transforms.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from pix2pix_turbo_from_cyclegan import Pix2Pix_Turbo_from_CycleGAN
from cyclegan_turbo import CycleGAN_Turbo

# ── layout constants ──────────────────────────────────────────────────────────
INNER_GAP = 4    # gap between columns within the same group (px)
GROUP_GAP = 18   # gap between reference / semantic / non-semantic groups (px)
ROW_GAP   = 4    # gap between the two image rows (px)


# ── image helpers ─────────────────────────────────────────────────────────────

def load_and_resize(path: str, resolution: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(resolution),
    ])(img)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """[0, 1] — for Pix2Pix_Turbo_from_CycleGAN (trained with this range)."""
    return F.to_tensor(img).unsqueeze(0).cuda()


def pil_to_cyclegan_tensor(img: Image.Image) -> torch.Tensor:
    """[-1, 1] — for CycleGAN_Turbo (trained with this range)."""
    x = F.to_tensor(img)
    x = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(x)
    return x.unsqueeze(0).cuda()


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    return transforms.ToPILImage()((t[0].cpu() * 0.5 + 0.5).clamp(0, 1))


def get_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def labelled_cell(img: Image.Image, text: str, font_size: int,
                  label_bg=(30, 30, 30), label_fg=(255, 255, 255)) -> Image.Image:
    """Return *img* with a thin coloured label strip at the top."""
    font   = get_font(font_size)
    bar_h  = font_size + 10
    cell   = Image.new("RGB", (img.width, img.height + bar_h), label_bg)
    cell.paste(img, (0, bar_h))
    draw   = ImageDraw.Draw(cell)
    bb     = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    draw.text(((img.width - tw) // 2, (bar_h - th) // 2), text, fill=label_fg, font=font)
    return cell


# ── row assembly ──────────────────────────────────────────────────────────────

def assemble_row(panels: list, group_sizes: list):
    """
    Place panels side-by-side with INNER_GAP inside each group and GROUP_GAP
    between groups.  group_sizes = [1, n_sem, n_nonsem, n_noft, ...] (ref always 1).
    Returns (row_image, x_offsets, col_widths).
    """
    widths = [p.width for p in panels]
    height = max(p.height for p in panels)

    # Build a set of indices that are the last panel in their group
    last_in_group = set()
    idx = 0
    for sz in group_sizes:
        if sz > 0:
            last_in_group.add(idx + sz - 1)
        idx += sz

    xs: list = []
    x = 0
    for i, w in enumerate(widths):
        xs.append(x)
        if i == len(panels) - 1:
            break
        gap = GROUP_GAP if i in last_in_group else INNER_GAP
        x += w + gap

    total_w = xs[-1] + widths[-1]
    row_img = Image.new("RGB", (total_w, height), (60, 60, 60))
    for panel, px in zip(panels, xs):
        row_img.paste(panel, (px, 0))
    return row_img, xs, widths


# ── group header ──────────────────────────────────────────────────────────────

def build_group_header(total_w: int, xs: list, widths: list,
                       group_sizes: list, group_specs: list, font_size: int) -> Image.Image:
    """
    Coloured banner spanning each non-reference group.
    group_sizes = [1, n_sem, n_nonsem, n_noft, ...]  (ref group first, no label drawn for it)
    group_specs = [(label, bg, fg), ...]              one entry per non-ref group (skip if size==0)
    """
    bar_h  = font_size + 14
    header = Image.new("RGB", (total_w, bar_h), (20, 20, 20))
    draw   = ImageDraw.Draw(header)
    font   = get_font(font_size)

    def draw_block(x0, x1, bg, fg, label):
        draw.rectangle([x0, 2, x1 - 1, bar_h - 3], fill=bg)
        bb = draw.textbbox((0, 0), label, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        w = x1 - x0
        draw.text((x0 + (w - tw) // 2, (bar_h - th) // 2), label, fill=fg, font=font)

    col_idx = group_sizes[0]  # skip ref group
    for (label, bg, fg), sz in zip(group_specs, group_sizes[1:]):
        if sz > 0:
            x0 = xs[col_idx]
            x1 = xs[col_idx + sz - 1] + widths[col_idx + sz - 1]
            draw_block(x0, x1, bg, fg, label)
        col_idx += sz

    return header


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_finetuned(model: Pix2Pix_Turbo_from_CycleGAN,
                  img: Image.Image, prompt: str) -> Image.Image:
    """Pix2Pix fine-tuned: expects [0,1] input."""
    out = model(pil_to_tensor(img), prompt=prompt)
    return tensor_to_pil(out)


@torch.no_grad()
def run_cyclegan(model: CycleGAN_Turbo,
                 img: Image.Image, prompt: str) -> Image.Image:
    """Base CycleGAN: expects [-1,1] input, called with direction=a2b."""
    out = model(pil_to_cyclegan_tensor(img), direction="a2b", caption=prompt)
    return tensor_to_pil(out)


def load_finetuned(path: str, use_fp16: bool) -> Pix2Pix_Turbo_from_CycleGAN:
    m = Pix2Pix_Turbo_from_CycleGAN(cyclegan_checkpoint_path=path)
    m.set_eval()
    m.unet.enable_xformers_memory_efficient_attention()
    if use_fp16:
        m.half()
    return m


def load_cyclegan_model(path: str, use_fp16: bool) -> CycleGAN_Turbo:
    m = CycleGAN_Turbo(pretrained_path=path)
    m.eval()
    m.unet.enable_xformers_memory_efficient_attention()
    if use_fp16:
        m.half()
    return m


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Multi-model ablation comparison grid.")
    ap.add_argument("--data_dir", required=True,
                    help="Root folder: test_A/, test_A_Masked/, test_prompts.json, fixed_prompt_b.txt")
    ap.add_argument("--config", required=True,
                    help="JSON config with 'semantic' and 'nonsemantic' model lists")
    ap.add_argument("--output_dir", default="./AllModelsComparison")
    ap.add_argument("--resolution", type=int, default=512,
                    help="Resize all images to this size before inference (default: 512)")
    ap.add_argument("--prompt", default=None,
                    help="Override ALL prompts for every model (semantic and non-semantic alike)")
    ap.add_argument("--fixed_prompt", default=None,
                    help="Prompt for non-semantic models (falls back to fixed_prompt_b.txt)")
    ap.add_argument("--use_fp16", action="store_true",
                    help="Use FP16 for faster inference")
    return ap.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    dir_orig     = os.path.join(args.data_dir, "test_A")
    dir_masked   = os.path.join(args.data_dir, "test_A_Masked")
    prompts_file = os.path.join(args.data_dir, "test_prompts.json")

    for d in (dir_orig, dir_masked):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Directory not found: {d}")
    if not os.path.isfile(prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    with open(prompts_file) as fh:
        prompts: dict = json.load(fh)

    # Load model config
    with open(args.config) as fh:
        config = json.load(fh)

    sem_cfgs    = config.get("semantic",    [])
    nonsem_cfgs = config.get("nonsemantic", [])
    noft_cfgs   = config.get("nofinetune",  [])
    n_sem, n_nonsem, n_noft = len(sem_cfgs), len(nonsem_cfgs), len(noft_cfgs)

    if n_sem + n_nonsem + n_noft == 0:
        raise ValueError("No models found in config JSON.")

    # --prompt overrides everything; otherwise fall back to per-group prompt sources
    global_prompt = args.prompt
    if global_prompt is not None:
        fixed_prompt = global_prompt
        print(f"Global prompt override: '{global_prompt}'")
    else:
        # Fixed prompt — only required when non-semantic or no-fine-tune models are present
        fixed_prompt = args.fixed_prompt
        if fixed_prompt is None and (n_nonsem > 0 or n_noft > 0):
            fp_path = os.path.join(args.data_dir, "fixed_prompt_b.txt")
            if not os.path.isfile(fp_path):
                raise FileNotFoundError(
                    "--fixed_prompt not given and fixed_prompt_b.txt not found in data_dir "
                    "(required for non-semantic / no-fine-tune models)"
                )
            with open(fp_path) as fh:
                fixed_prompt = fh.read().strip()
            print(f"Fixed prompt (non-semantic): '{fixed_prompt}'")

    print(f"Loading {n_sem} semantic + {n_nonsem} non-semantic + {n_noft} no-fine-tune models …")
    sem_models    = [(c["name"], load_finetuned(c["path"],      args.use_fp16)) for c in sem_cfgs]
    nonsem_models = [(c["name"], load_finetuned(c["path"],      args.use_fp16)) for c in nonsem_cfgs]
    noft_models   = [(c["name"], load_cyclegan_model(c["path"], args.use_fp16)) for c in noft_cfgs]

    os.makedirs(args.output_dir, exist_ok=True)

    supported   = (".png", ".jpg", ".jpeg", ".webp")
    image_names = sorted(f for f in os.listdir(dir_orig) if f.lower().endswith(supported))
    if not image_names:
        raise RuntimeError(f"No images found in {dir_orig}")

    # Scale font with resolution so labels stay legible
    font_size = max(10, args.resolution // 18)

    # Label bar colours (bg, fg) per column group
    REF_LBG   = (30, 30, 30)
    SEM_LBG   = (28, 45, 75)    # blue tint  — semantic fine-tuned
    NSEM_LBG  = (75, 28, 28)    # red tint   — non-semantic fine-tuned
    NOFT_LBG  = (28, 65, 40)    # green tint — no fine-tune (base CycleGAN)
    WHITE_FG  = (255, 255, 255)

    # Group specs for the header banner: (label, bg, fg) — one per non-ref group
    GROUP_SPECS = [
        ("SEMANTIC",       (30,  55, 110), (160, 215, 255)),
        ("NON-SEMANTIC",   (110, 30,  30), (255, 190, 160)),
        ("NO FINE-TUNE",   (28,  90,  55), (160, 255, 200)),
    ]
    GROUP_SIZES = [1, n_sem, n_nonsem, n_noft]

    print(f"Preprocessing: resize to {args.resolution}px (LANCZOS)")
    print(f"{len(image_names)} images — running inference …")

    for fname in image_names:
        stem        = os.path.splitext(fname)[0]
        path_orig   = os.path.join(dir_orig,   fname)
        path_masked = os.path.join(dir_masked, fname)

        if not os.path.isfile(path_masked):
            print(f"  [SKIP] no masked image for {fname}")
            continue

        if global_prompt is not None:
            prompt = global_prompt
        else:
            prompt = prompts.get(fname) or prompts.get(stem)
            if prompt is None:
                print(f"  [SKIP] no prompt found for {fname}")
                continue

        print(f"  {fname}")

        img_orig   = load_and_resize(path_orig,   args.resolution)
        img_masked = load_and_resize(path_masked, args.resolution)

        # ── inference ─────────────────────────────────────────────────────────
        sem_out_masked  = [(n, run_finetuned(m, img_masked, prompt))       for n, m in sem_models]
        sem_out_orig    = [(n, run_finetuned(m, img_orig,   prompt))       for n, m in sem_models]
        nsem_out_masked = [(n, run_finetuned(m, img_masked, fixed_prompt)) for n, m in nonsem_models]
        nsem_out_orig   = [(n, run_finetuned(m, img_orig,   fixed_prompt)) for n, m in nonsem_models]
        # Base CycleGAN uses [-1,1] input and the fixed prompt (no per-image conditioning)
        noft_out_masked = [(n, run_cyclegan(m, img_masked, fixed_prompt))  for n, m in noft_models]
        noft_out_orig   = [(n, run_cyclegan(m, img_orig,   fixed_prompt))  for n, m in noft_models]

        # ── build panel rows ──────────────────────────────────────────────────
        def make_panels(ref_img, ref_lbl, sem_pairs, nsem_pairs, noft_pairs):
            panels = [labelled_cell(ref_img, ref_lbl, font_size, REF_LBG, WHITE_FG)]
            for name, out in sem_pairs:
                panels.append(labelled_cell(out, name, font_size, SEM_LBG,  WHITE_FG))
            for name, out in nsem_pairs:
                panels.append(labelled_cell(out, name, font_size, NSEM_LBG, WHITE_FG))
            for name, out in noft_pairs:
                panels.append(labelled_cell(out, name, font_size, NOFT_LBG, WHITE_FG))
            return panels

        panels_masked = make_panels(img_masked, "Ref (masked)",
                                    sem_out_masked,  nsem_out_masked,  noft_out_masked)
        panels_orig   = make_panels(img_orig,   "Ref (original)",
                                    sem_out_orig,    nsem_out_orig,    noft_out_orig)

        row1, xs, col_widths = assemble_row(panels_masked, GROUP_SIZES)
        row2, _,  _          = assemble_row(panels_orig,   GROUP_SIZES)

        # ── stack rows ────────────────────────────────────────────────────────
        body_h = row1.height + ROW_GAP + row2.height
        body   = Image.new("RGB", (row1.width, body_h), (60, 60, 60))
        body.paste(row1, (0, 0))
        body.paste(row2, (0, row1.height + ROW_GAP))

        # ── group header bar (only when more than one model group is present) ──
        n_active_groups = sum(1 for s in GROUP_SIZES[1:] if s > 0)
        if n_active_groups > 1:
            header = build_group_header(row1.width, xs, col_widths,
                                        GROUP_SIZES, GROUP_SPECS, font_size + 2)
            final_h = header.height + ROW_GAP + body.height
            final   = Image.new("RGB", (row1.width, final_h), (60, 60, 60))
            final.paste(header, (0, 0))
            final.paste(body,   (0, header.height + ROW_GAP))
        else:
            final = body

        out_path = os.path.join(args.output_dir, f"{stem}_ablation.png")
        final.save(out_path)
        print(f"    → {out_path}")

    print(f"\nDone. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()


"""
First we do finetunings:

python compare_all_models.py --data_dir /data/upftfg19/mfsvensson/Data_TFG/dataToComparePaired --config ../compare_models_dirs/models_config_1001.json --output_dir ../FineTuning_1001
python compare_all_models.py --data_dir /data/upftfg19/mfsvensson/Data_TFG/dataToComparePaired --config ../compare_models_dirs/models_config_3001.json --output_dir ../FineTuning_3001
python compare_all_models.py --data_dir /data/upftfg19/mfsvensson/Data_TFG/dataToComparePaired --config ../compare_models_dirs/models_config_5001.json --output_dir ../FineTuning_5001

Normal CycleGAN (no fine-tuning) for reference:

python compare_all_models.py --data_dir /data/upftfg19/mfsvensson/Data_TFG/dataToCompareUnpaired --config ../compare_models_dirs/cycle_test.json --output_dir ../CycleGan_Test --prompt "Generate a detailed, realistic, highly textured, and full-color illustration imgae."
"""