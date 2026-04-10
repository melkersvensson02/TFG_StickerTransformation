#!/bin/bash
#SBATCH --job-name=cycle_turbo_skip1
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
OUTDIR="/home/mfsvensson/TFG_reps/Old-img2img-turbo/outputs/baseLine_LongTrained_Cycle"

RES=512
BS=2
MAX_STEPS=4000

echo "Checking if Python file exists:"
ls -l src/cyclegan_train_turbo.py

echo "Starting training with skip connections enabled (weight = 1.0)"

# ----------------------------------------------------------
# Run training
# ----------------------------------------------------------

accelerate launch src/cyclegan_train_turbo.py \
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
    --vae_skip_weight 1.0 --validation_num_images 75 \

echo "Training completed."
