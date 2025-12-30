#!/bin/bash
#SBATCH --job-name=pix2pix_turbo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ----------------------------------------------------------
# Environment setup
# ----------------------------------------------------------

cd /home/mfsvensson/TFG_reps/Old-img2img-turbo
echo "Running from PWD: $(pwd)"

# Load the module system's conda module (required on your cluster)
module load conda

# Activate your environment
source activate Pined-img2img-turbo-fixed   # <<< change this to your actual env name

# (Optional but useful)
echo "Running on host:" $(hostname)
echo "CUDA devices available:"
nvidia-smi

# ----------------------------------------------------------
# Run training
# ----------------------------------------------------------

echo "Checking if Python file exists:"
ls -l src/Train_pix2pix_turbo.py


echo "We are about to run the python code!"
BASE_MODEL="/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo"
DATASET="/data/upftfg19/mfsvensson/Data_TFG/PairedDatasetSemantic"
RES=512
BS=4
LAMBDA_L2=1.0
LAMBDA_GAN=1.0
LAMBDA_CLIPSIM=0.0
LPIPS=0.0
CANNY_LOW=100
CANNY_HIGH=200

OUTDIR="/home/mfsvensson/TFG_reps/Old-img2img-turbo/outputs/best_ablation_original_v1/ML2_G_B_skip_0.5_canny_finetuned"
MAX_STEEPS=5510 # five times more the original 1102
SKIP_ARGS="--skip_weight 0.5"
COND_ARGS="--use_canny_conditioning True --canny_low_threshold ${CANNY_LOW} --canny_high_threshold ${CANNY_HIGH}"

accelerate launch src/Train_pix2pix_turbo.py \
    --pretrained_model_name_or_path="${BASE_MODEL}" \
    --output_dir="${OUTDIR}" \
    --dataset_folder="${DATASET}" \
    --resolution="${RES}" \
    --train_batch_size "${BS}" \
    --max_train_steps "${MAX_STEPS}" \
    --enable_xformers_memory_efficient_attention \
    --viz_freq 25 \
    --track_val_fid \
    --mixed_precision="bf16" \
    --num_samples_eval 100 \
    --data_set_type "modified" \
    --checkpointing_steps 1000 \
    --lambda_clipsim "${LAMBDA_CLIPSIM}" \
    --lambda_lpips "${LPIPS}" \
    --lambda_gan "${LAMBDA_GAN}" \
    --lambda_l2 "${LAMBDA_L2}" \
    ${SKIP_ARGS} \
    ${COND_ARGS}