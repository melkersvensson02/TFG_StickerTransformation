#!/bin/bash
#SBATCH --job-name=ablation_gdl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ----------------------------------------------------------
# Environment setup
# ----------------------------------------------------------

cd /home/mfsvensson/TFG_reps/Old-img2img-turbo
echo "Running from PWD: $(pwd)"

module load conda
source activate Pined-img2img-turbo-fixed

echo "Running on host: $(hostname)"
echo "CUDA devices available:"
nvidia-smi

echo "GPUs allocated to this job: $CUDA_VISIBLE_DEVICES"
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Number of GPUs available to job: $NUM_GPUS"

# ----------------------------------------------------------
# Paths and parameters
# ----------------------------------------------------------

BASE_MODEL="/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo"
DATASET="/data/upftfg19/mfsvensson/Data_TFG/Final_unpaired"
OUTDIR="/home/mfsvensson/TFG_reps/Old-img2img-turbo/outputs/ablation_4_gdl"

BS=2
MAX_STEPS=6002

echo "Checking if Python file exists:"
ls -l src/Cyclegan_Train_turbo.py

echo "Starting training: Default but λ_cycle = 0.5, λ_cycle_lpips = 10.0, λ_gdl = 0.5"

# ----------------------------------------------------------
# Run training
# ----------------------------------------------------------

accelerate launch src/Cyclegan_Train_turbo.py \
    --pretrained_model_name_or_path "${BASE_MODEL}" \
    --output_dir "${OUTDIR}" \
    --dataset_folder "${DATASET}" \
    --train_batch_size "${BS}" \
    --max_train_steps "${MAX_STEPS}" \
    --enable_xformers_memory_efficient_attention \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --train_img_prep "resize_512" \
    --val_img_prep "resize_512" \
    --viz_freq 25 \
    --vae_ignore_skip False \
    --vae_skip_weight 1.0 \
    --validation_num_images 75 \
    --lambda_gan 0.5 \
    --lambda_cycle 0.5 \
    --lambda_cycle_lpips 10.0 \
    --lambda_idt 0.5 \
    --lambda_idt_lpips 1.0 \
    --lambda_gdl 0.5 \
    --lambda_cx 0.0 \ 
    --mixed_precision "fp16" \

echo "Training completed."
