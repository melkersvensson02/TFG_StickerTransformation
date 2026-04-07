# Fine-tuning CycleGAN with Pix2Pix Setup

This guide explains how to fine-tune a trained CycleGAN model using paired data with pix2pix-style losses.

## Overview

After training a CycleGAN model on unpaired data, you can fine-tune it using paired data to improve quality. This approach:

1. **Loads your trained CycleGAN checkpoint** - preserves learned transformations
2. **Uses paired dataset** - leverages aligned input-output pairs
3. **Applies pix2pix losses** - L2, LPIPS, CLIP-sim, and GAN discriminator
4. **Supports flexible prompting** - per-image semantic prompts OR generic transformation prompt
5. **Saves compatible checkpoints** - can be used with `cyclegan_turbo.py` for inference

## New Files Created

### 1. `pix2pix_turbo_from_cyclegan.py`

A model class that loads CycleGAN checkpoints for fine-tuning.

**Key features:**
- Uses 3 LoRA adapters for UNet (encoder/decoder/others) matching CycleGAN structure
- Single-direction VAE (a2b only: photo → sticker)
- Loads and saves checkpoints compatible with `cyclegan_turbo.py`
- Includes diagnostic prints for debugging weight loading

**Usage in code:**
```python
from pix2pix_turbo_from_cyclegan import Pix2Pix_Turbo_from_CycleGAN

model = Pix2Pix_Turbo_from_CycleGAN(
    cyclegan_checkpoint_path="outputs/cyclegan/checkpoints/model_5000.pkl",
    lora_rank_unet=8,
    lora_rank_vae=4,
)
```

### 2. `train_pix2pix_turbo_from_cyclegan.py`

Training script for fine-tuning with paired data.

**Key features:**
- Based on `train_pix2pix_turbo.py` (same training loop structure)
- Loads CycleGAN checkpoint at initialization
- Uses standard L2 loss (better for domain displacement)
- Supports both prompt strategies via `--data_set_type` flag
- Fresh discriminator (starts from scratch)

## Dataset Requirements

### Paired Dataset Structure

Your dataset should be organized as:

```
dataset_folder/
├── train_A/          # Input images (photos)
├── train_B/          # Target images (stickers)
├── test_A/           # Test input images
├── test_B/           # Test target images
├── train_prompts.json   # Optional: per-image prompts for training
└── test_prompts.json    # Optional: per-image prompts for testing
```

### Two Prompting Strategies

#### Option 1: Per-Image Semantic Prompts (`data_set_type="original"`)
Uses individual captions for each image via JSON files.

**train_prompts.json format:**
json
{
    "image001.png": "a portrait of a person with glasses",
    "image002.png": "a dog sitting in grass",
    "image003.png": "a cityscape at sunset"
}


#### Option 2: Generic Transformation Prompt (`data_set_type="modified"`)
Uses a single hardcoded prompt for all images (similar to CycleGAN training).

**Current generic prompt** (in `PairedDataset` class):
```python
"Minimal black tattoo line art, bold clean contour lines, simplified details, sticker-style, high contrast."
```

**To customize:** Edit line 418-420 in `src/my_utils/training_utils.py`:
```python
self.caption = (
    "YOUR CUSTOM TRANSFORMATION DESCRIPTION HERE"
)
```

## Training Commands

### Example 1: Fine-tune with Per-Image Prompts

```bash
accelerate launch src/train_pix2pix_turbo_from_cyclegan.py \
    --cyclegan_checkpoint="outputs/cyclegan_run/checkpoints/model_5000.pkl" \
    --output_dir="outputs/finetuned_with_prompts/" \
    --dataset_folder="/path/to/your/paired_dataset" \
    --data_set_type="original" \
    --resolution=512 \
    --train_batch_size=16 \
    --max_train_steps=10000 \
    --checkpointing_steps=500 \
    --eval_freq=500 \
    --enable_xformers_memory_efficient_attention \
    --track_val_fid \
    --dataloader_num_workers=4 \
    --mixed_precision="bf16" \
    --learning_rate=5e-6 \
    --lambda_l2=1.0 \
    --lambda_lpips=5.0 \
    --lambda_clipsim=5.0 \
    --lambda_gan=0.5
```

### Example 2: Fine-tune with Generic Prompt

```bash
accelerate launch src/train_pix2pix_turbo_from_cyclegan.py \
    --cyclegan_checkpoint="outputs/cyclegan_run/checkpoints/model_5000.pkl" \
    --output_dir="outputs/finetuned_generic/" \
    --dataset_folder="/path/to/your/paired_dataset" \
    --data_set_type="modified" \
    --resolution=512 \
    --train_batch_size=16 \
    --max_train_steps=10000 \
    --checkpointing_steps=500 \
    --eval_freq=500 \
    --enable_xformers_memory_efficient_attention \
    --track_val_fid \
    --dataloader_num_workers=4 \
    --mixed_precision="bf16"
```

