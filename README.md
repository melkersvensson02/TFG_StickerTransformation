# Domain Specific Img2Img Transformation
**Bachelor's Thesis — Melker Fredrik Svensson**
Supervisors: Pablo Arias, Gloria Haro · UPF Computer Engineering, 2025–2026

---

## Overview

This project fine-tunes **SD-Turbo** for a domain-specific image-to-image translation task: given an RGB photograph, generate a **binary sticker representation** of the main subjects (outline-style, no background). Training is studied in both **paired** (Pix2Pix-Turbo) and **unpaired** (CycleGAN-Turbo) settings, using **LoRA** adapters on the UNet and VAE.

The paired setup proved insufficient — pixel-level losses could not overcome the spatial displacement and hallucinations present in the generated dataset. The unpaired cycle-consistency framework significantly outperformed it. The best results were obtained by first training CycleGAN-Turbo on a segmented unpaired dataset, then fine-tuning with paired pix2pix losses.

Key contributions:
- A novel paired/unpaired sticker dataset generated via Gemini 2.5 Flash + Grounded SAM 2
- Architecture modifications: controllable skip-connection weight (`alpha_skip`)
- Custom loss functions: **Masked L2** (upweights black-pixel errors) and **Contextual Loss** (spatially invariant feature matching)
- Extensive ablation studies across both training settings

---

## Project Structure

```
src/
  cyclegan_turbo.py               # CycleGAN_Turbo model definition
  cyclegan_train_turbo.py         # CycleGAN unpaired training script
  pix2pix_turbo.py                # Pix2Pix_Turbo model definition
  train_pix2pix_turbo.py          # Pix2Pix paired training script
  pix2pix_turbo_from_cyclegan.py  # Pix2Pix model initialized from CycleGAN checkpoint
  train_pix2pix_turbo_from_cyclegan.py  # Fine-tune CycleGAN with pix2pix losses
  inference_unpaired.py           # CycleGAN inference (single image)
  inference_paired.py             # Pix2Pix inference (single image, Canny-guided)
  compare_all_models.py           # Side-by-side grid comparison of all model types
  compare_pix2pix_models.py       # Side-by-side grid comparison of pix2pix checkpoints
  visualize_cycle_grid.py         # 2x3 cycle-consistency grid visualization
  cycle_test.py                   # Evaluate CycleGAN on a test set
  model.py                        # Shared utilities (scheduler, VAE helpers)
jobs/                             # SLURM job scripts for HPC training
checkpoints/                      # Saved model checkpoints
dataset/                          # Paired and unpaired training data
```

---

## Setup

### 1. Create the conda environment

```bash
conda env create -f Pined.yaml
conda activate Pined-img2img-turbo-fixed
```

### 2. Additional dependencies (install manually if needed)

```bash
pip install clip-by-openai accelerate vision-aided-loss wandb
```

### 3. Weights

Pre-trained SD-Turbo weights are loaded from HuggingFace automatically. Custom checkpoints are stored in `checkpoints/` and `ckpt_folder/`.
Go to https://huggingface.co/stabilityai/sd-turbo/tree/main to get the weights for SD-Turbo!

---

## Training

### Unpaired — CycleGAN-Turbo

```bash
accelerate launch src/cyclegan_train_turbo.py \
  --dataset_folder /path/to/unpaired_dataset \
  --prompt_a "a photo of a subject" \
  --prompt_b "a sticker of a subject" \
  --output_dir ./outputs/cyclegan_run \
  --max_train_steps 6000 \
  --train_batch_size 1
```

The dataset folder must contain `train_A/` and `train_B/` subdirectories. For the sticker task, `train_A` should contain segmented RGB images and `train_B` sticker images.

---

### Paired fine-tuning from a CycleGAN checkpoint

```bash
accelerate launch src/train_pix2pix_turbo_from_cyclegan.py \
  --pretrained_model_path /path/to/cyclegan_checkpoint.pkl \
  --dataset_folder /path/to/paired_dataset \
  --output_dir ./outputs/finetuned_run \
  --max_train_steps 5001 \
  --lambda_lpips 5.0 --lambda_l2 1.0 --lambda_clipsim 5.0 --lambda_gan 0.5
```

---

### Paired — Pix2Pix-Turbo (from scratch)

```bash
accelerate launch src/train_pix2pix_turbo.py \
  --dataset_folder /path/to/paired_dataset \
  --output_dir ./outputs/pix2pix_run \
  --max_train_steps 3000
```

---

## Inference

### CycleGAN (unpaired model)

```bash
python src/inference_unpaired.py \
  --input_image /path/to/image.jpg \
  --model_path /path/to/checkpoint.pkl \
  --prompt "a sticker of a dog" \
  --direction a2b \
  --output_dir ./output
```

### Pix2Pix

```bash
python src/inference_paired.py \
  --input_image /path/to/image.jpg \
  --model_path /path/to/checkpoint.pkl \
  --prompt "a sticker of a dog"
```

---

## Evaluation & Visualization

```bash
# Cycle-consistency grid (2x3)
python src/visualize_cycle_grid.py \
  --model_path /path/to/checkpoint.pkl \
  --dataset_folder /path/to/test_data \
  --output_dir ./cycle_grids

# Compare all model types side-by-side
python src/compare_all_models.py \
  --data_dir /path/to/test_data \
  --config compare_models_dirs/models_config_5001.json \
  --output_dir ./comparison_output
```

---

## Dependencies (`Pined.yaml`)

```yaml
name: Pined-img2img-turbo-fixed
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch=2.3.1
  - torchvision=0.18.1
  - torchaudio=2.3.1
  - pytorch-cuda=12.1
  - pip:
      - diffusers==0.25.1
      - peft==0.7.1
      - transformers==4.35.2
      - accelerate
      - lpips==0.1.4
      - clean-fid==0.1.35
      - open-clip-torch==2.20.0
      - opencv-python==4.6.0.66
      - pillow==9.5.0
      - scipy==1.11.1
      - timm==0.9.2
      - tqdm==4.65.0
      - huggingface-hub==0.20.3
      - gradio==3.43.1
      - dominate==2.9.1
      - numpy==1.24.4
```

Full pinned spec: [`Pined.yaml`](Pined.yaml)

---

## Reference

This work is based on:
> Parmar et al., *One-Step Image Translation with Text-to-Image Models*, CVPR 2024.
> [[img2img-turbo](https://github.com/GaParmar/img2img-turbo)]
