#!/bin/bash
#SBATCH --job-name=cycle_turbo_skip1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
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

echo "Checking if Python file exists:"
ls -l src/cyclegan_train_turbo.py

echo "Starting training with skip connections enabled (weight = 1.0)"
export NCCL_P2P_DISABLE=1
accelerate launch src/Cyclegan_Train_turbo_resume.py \
    --pretrained_model_name_or_path="/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo" \
    --output_dir=OUTDIR="/home/mfsvensson/TFG_reps/Old-img2img-turbo/outputs/baseLine_LongTrained_Cycle_resume" \
    --dataset_folder "/data/upftfg19/mfsvensson/Data_TFG/Final_unpaired" \
    --train_img_prep "resize_512" \
    --val_img_prep "no_resize" \
    --learning_rate="1e-5" \
    --max_train_steps=8000 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --enable_xformers_memory_efficient_attention \
    --validation_steps 75 \
    --mixed_precision bf16 \
    --resume_from_checkpoint="/home/mfsvensson/TFG_reps/Old-img2img-turbo/outputs/baseLine_LongTrained_Cycle/checkpoints/model_4001.pkl" \