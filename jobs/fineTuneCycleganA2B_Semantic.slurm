#!/bin/bash
#SBATCH --job-name=fineTuneCycleganA2B_Semantic
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

BASE_MODEL="/home/mfsvensson/TFG_reps/Old-img2img-turbo/outputs/many_runs_bs_1/checkpoints/model_6001.pkl"
DATASET="/data/upftfg19/mfsvensson/Data_TFG/NewPairedDatasetSemantic"
OUTDIR="/home/mfsvensson/TFG_reps/Old-img2img-turbo/outputs/fineTune_cyclegan_a2b_semantic"

BS=2
MAX_STEPS=6002

echo "Checking if Python file exists:"
ls -l src/train_pix2pix_turbo_from_cyclegan.py

# ----------------------------------------------------------
# Run training
# The reason is because CLIP_sim is a semantic loss and here we use a generic prompt about the transition !
# ----------------------------------------------------------
accelerate launch src/train_pix2pix_turbo_from_cyclegan.py \
    --cyclegan_checkpoint="$BASE_MODEL" \
    --output_dir="$OUTDIR" \
    --dataset_folder="$DATASET" \
    --data_set_type="original" \
    --resolution=512 \
    --train_batch_size=1 \
    --enable_xformers_memory_efficient_attention \
    --track_val_fid \
    --max_train_steps=5002 \
    --checkpointing_steps=1000 \
    --eval_freq=500 \
    --viz_freq=25 \

