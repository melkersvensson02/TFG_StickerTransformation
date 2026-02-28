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
```json
{
    "image001.png": "a portrait of a person with glasses",
    "image002.png": "a dog sitting in grass",
    "image003.png": "a cityscape at sunset"
}
```

**test_prompts.json format:**
```json
{
    "test001.png": "a cat on a windowsill",
    "test002.png": "a mountain landscape"
}
```

**Advantages:**
- More semantic guidance per image
- Better for diverse datasets with varied content
- Can guide style/content more precisely

#### Option 2: Generic Transformation Prompt (`data_set_type="modified"`)

Uses a single hardcoded prompt for all images (similar to CycleGAN training).

**Current generic prompt** (in `PairedDataset` class):
```python
"Minimal black tattoo line art, bold clean contour lines, simplified details, sticker-style, high contrast."
```

**Advantages:**
- Simpler setup (no JSON files needed)
- Focuses on transformation rather than content
- Consistent with CycleGAN training approach

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

## Key Arguments Explained

### Required Arguments

- `--cyclegan_checkpoint`: Path to your trained CycleGAN .pkl file
- `--output_dir`: Where to save checkpoints and logs
- `--dataset_folder`: Root folder of paired dataset

### Prompting Strategy

- `--data_set_type`: Choose `"original"` (per-image prompts) or `"modified"` (generic prompt)

### Loss Weights

- `--lambda_l2`: L2 reconstruction loss weight (default: 1.0)
- `--lambda_lpips`: Perceptual loss weight (default: 5.0)
- `--lambda_clipsim`: CLIP similarity loss weight (default: 5.0)
- `--lambda_gan`: GAN discriminator loss weight (default: 0.5)

**Note:** Standard L2 loss is used (not weighted by ink) because target domain displacement makes weighted loss less effective.

### Training Schedule

- `--max_train_steps`: Total training steps (e.g., 10000)
- `--checkpointing_steps`: Save checkpoint every N steps (e.g., 500)
- `--eval_freq`: Evaluate metrics every N steps (e.g., 500)

### Performance

- `--train_batch_size`: Batch size per GPU (e.g., 16)
- `--mixed_precision`: Use `"bf16"` for Ampere GPUs or `"fp16"` for older GPUs
- `--enable_xformers_memory_efficient_attention`: Reduces memory usage
- `--gradient_checkpointing`: Further reduces memory (slower training)

### Validation

- `--track_val_fid`: Compute FID score during validation
- `--num_samples_eval`: Number of validation images to use (default: 100)

## Output Files

Training produces the following outputs:

```
output_dir/
├── checkpoints/
│   ├── model_step_500.pkl      # Checkpoint at step 500
│   ├── model_step_1000.pkl     # Checkpoint at step 1000
│   └── ...
├── eval/
│   ├── fid_500/                # Generated images for FID at step 500
│   └── fid_1000/
├── debug_dataloader/           # Sample images from dataloaders
│   ├── train_batch1_A_0.png
│   ├── train_batch1_B_0.png
│   └── ...
├── debug_train_outputs/        # Model outputs during training
│   ├── step_500_A.png          # Input
│   ├── step_500_B.png          # Output
│   └── ...
├── metrics.jsonl               # Training metrics (JSON lines)
└── running_information.txt     # Training log
```

## Checkpoint Compatibility

Checkpoints saved during fine-tuning use **CycleGAN format** and can be loaded by:

1. **For inference:** Use `cyclegan_turbo.py`
   ```python
   from cyclegan_turbo import CycleGAN_Turbo
   
   model = CycleGAN_Turbo(pretrained_path="outputs/finetuned/checkpoints/model_step_5000.pkl")
   model.eval()
   
   # Run inference
   output = model(input_image, direction="a2b")
   ```

2. **For further fine-tuning:** Use `train_pix2pix_turbo_from_cyclegan.py` again
   ```bash
   accelerate launch src/train_pix2pix_turbo_from_cyclegan.py \
       --cyclegan_checkpoint="outputs/finetuned/checkpoints/model_step_5000.pkl" \
       ...
   ```

## Monitoring Training

### Weights & Biases (W&B)

If using `--report_to="wandb"`, metrics are logged to W&B:
- Training losses: L2, LPIPS, CLIP-sim, Generator, Discriminator
- Validation metrics: L2, LPIPS, CLIP-sim, FID
- Learning rate curves

### Local Logs

- **metrics.jsonl**: JSON lines format, one entry per evaluation
  ```json
  {"step": 500, "train_avg_l2": 0.123, "val/l2": 0.098, "val/clean_fid": 45.2, ...}
  ```

- **running_information.txt**: Human-readable training log with:
  - Model initialization details
  - Dataset statistics
  - VAE skip layer verification
  - Training progress

## Technical Details

### How LoRA Adapters Work

CycleGAN uses **3 separate LoRA adapters** for the UNet:

1. **default_encoder** - LoRA weights for down_blocks + conv_in
2. **default_decoder** - LoRA weights for up_blocks
3. **default_others** - LoRA weights for mid_block and other modules

These adapter names act as dictionary keys when saving/loading:

```python
# Saving
sd = {
    "sd_encoder": {...},    # Weights for default_encoder
    "sd_decoder": {...},    # Weights for default_decoder
    "sd_other": {...},      # Weights for default_others
}

# Loading
for n, p in unet.named_parameters():
    if "default_encoder" in n:
        # Parameter name: "down_blocks.0.to_k.default_encoder.weight"
        # Saved key:      "down_blocks.0.to_k.weight"
        clean_name = n.replace(".default_encoder.weight", ".weight")
        p.data.copy_(sd["sd_encoder"][clean_name])
```

This structure ensures checkpoints remain compatible with `cyclegan_turbo.py`.

### VAE Direction Handling

- **CycleGAN training:** Uses bidirectional VAEs (a2b and b2a wrapped in `VAE_encode`/`VAE_decode`)
- **Fine-tuning:** Uses only **a2b direction** (simpler, single-direction)
- **Loading:** Extracts only `vae.` weights from checkpoint, ignoring `vae_b2a.` weights

### Discriminator

- **Starts from scratch** (not loaded from checkpoint)
- This is beneficial: fresh discriminator adapts to paired data distribution
- Uses Vision-Aided GAN with CLIP features

### Loss Function Differences

Compared to `train_pix2pix_turbo.py`:

1. **Standard L2 loss** instead of weighted L2
   - Weighted loss prioritizes ink/lines
   - Not effective when target domain has displacement
   
2. **Same LPIPS, CLIP-sim, GAN losses**

## Troubleshooting

### Issue: "Checkpoint missing required key"

**Cause:** Trying to load a non-CycleGAN checkpoint  
**Solution:** Ensure `--cyclegan_checkpoint` points to a .pkl file from CycleGAN training

### Issue: "VAE skip layers appear to be near zero"

**Cause:** Skip connection weights didn't load correctly  
**Solution:** Check the diagnostic output - if all skip_conv layers show sum < 1e-6, the checkpoint may be corrupted

### Issue: Out of memory

**Solutions:**
- Reduce `--train_batch_size` (try 8 or 4)
- Enable `--gradient_checkpointing`
- Use `--mixed_precision="bf16"`
- Reduce `--resolution` if possible

### Issue: FID score not improving

**Possible causes:**
- Learning rate too high/low (try 1e-6 to 1e-5)
- Loss weights need tuning
- Dataset quality issues
- Need more training steps

**Try:**
- Adjust `--lambda_lpips` and `--lambda_gan`
- Visualize outputs in `debug_train_outputs/`
- Check validation images in `eval/fid_*/`

## Recommended Workflow

1. **Train CycleGAN on unpaired data** (using `Cyclegan_Train_turbo.py`)
   - Get rough transformation working
   - Save checkpoint at convergence (e.g., step 5000)

2. **Fine-tune with paired data using per-image prompts**
   - Create train_prompts.json / test_prompts.json
   - Run with `--data_set_type="original"`
   - Train for ~5000-10000 steps
   - Monitor FID, LPIPS on validation set

3. **Optional: Try generic prompt approach**
   - Run with `--data_set_type="modified"`
   - Compare results with per-image prompts
   - Use whichever works better for your use case

4. **Inference with best checkpoint**
   - Use `cyclegan_turbo.py` or your inference script
   - Compare outputs before/after fine-tuning

## Citation

This implementation builds upon:
- [CycleGAN-Turbo](https://github.com/GaParmar/img2img-turbo) by Parmar et al.
- [Pix2Pix-Turbo](https://github.com/GaParmar/img2img-turbo) by Parmar et al.

If you use this code, please cite the original papers.
