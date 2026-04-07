#!/bin/bash
#SBATCH --job-name=new_cx_lpips_avg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

cd /home/mfsvensson/TFG_reps/Old-img2img-turbo
echo "Running from PWD: $(pwd)"

module load conda
source activate Pined-img2img-turbo-fixed

echo "Running on host: $(hostname)"
nvidia-smi
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

BASE_MODEL="/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo"
DATASET="/data/upftfg19/mfsvensson/Data_TFG/Final_unpaired"
OUTDIR="/home/mfsvensson/TFG_reps/Old-img2img-turbo/outputs/new_cx_lpips_avg"

# CX and LPIPS equally weighted in Lrec:
#   lambda_cycle_lpips = 5   (half the original weight)
#   lambda_idt_lpips   = 0.5 (half the original weight)
#   lambda_cx          = 5   (same weight as LPIPS → effective total ≈ original 10)
# Each contributes ~50% of the perceptual reconstruction signal.
# VGG is shared between CX and LPIPS (no extra GPU memory).

accelerate launch src/Cyclegan_Train_turbo.py \
    --pretrained_model_name_or_path "${BASE_MODEL}" \
    --output_dir "${OUTDIR}" \
    --dataset_folder "${DATASET}" \
    --train_batch_size 1 \
    --max_train_steps 6002 \
    --enable_xformers_memory_efficient_attention \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --train_img_prep "resize_512" \
    --val_img_prep "resize_512" \
    --validation_steps 250 \
    --validation_num_images 75 \
    --vae_ignore_skip False \
    --vae_skip_weight 1.0 \
    --lambda_gan 0.5 \
    --lambda_cycle 1.0 \
    --lambda_cycle_lpips 5.0 \
    --lambda_idt 1.0 \
    --lambda_idt_lpips 0.5 \
    --lambda_gdl 0.0 \
    --lambda_cx 5.0 \


echo "Training completed."
