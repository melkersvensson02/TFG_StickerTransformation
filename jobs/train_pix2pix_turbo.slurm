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
# -------------------------
# Base config
# -------------------------
BASE_MODEL="/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo"
DATASET="/data/upftfg19/mfsvensson/Data_TFG/PairedDatasetSemantic"

RES=512
BS=4
MAX_STEPS=1102

# Loss setup for ML2_G_B
LAMBDA_L2=1.0
LAMBDA_GAN=1.0
LAMBDA_CLIPSIM=0.0
LPIPS=0.0

# Conditioning defaults
CANNY_LOW=100
CANNY_HIGH=200
BIN_THR=0.55   # adjust if you want a different “logical threshold”

# -------------------------
# Grid (imp we use modified dataset in this case)
# -------------------------
for COND in normal canny; do
    for SKIP in 0.0 1.0 0.5; do
    # ----- output_dir naming -----
    OUTNAME="ML2_G_B_skip_${SKIP}_${COND}"
    if [[ "${LPIPS}" != "0.0" ]]; then
    OUTNAME="${OUTNAME}_lp015"
    fi
    OUTDIR="/home/mfsvensson/TFG_reps/Old-img2img-turbo/outputs/best_ablation_modified_v1/${OUTNAME}"

    # ----- skip args -----
    SKIP_ARGS=""
    if [[ "${SKIP}" == "0.0" ]]; then
    SKIP_ARGS="--ignore_skip True"
    else
    SKIP_ARGS="--skip_weight ${SKIP}"
    fi

    # ----- conditioning args ----- by deafult its False which would be normal
    if [[ "${COND}" == "canny" ]]; then
    COND_ARGS="--use_canny_conditioning True"
    COND_ARGS="${COND_ARGS} --canny_low_threshold ${CANNY_LOW} --canny_high_threshold ${CANNY_HIGH}"
    fi

    echo "Launching: ${OUTNAME}"

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

    done
done

# -------------------------
# This will produce the following output dirs:
# outputs/skip_ablation_v2/ML2_G_B_skip_0_normal
# outputs/skip_ablation_v2/ML2_G_B_skip_0.5_normal
# outputs/skip_ablation_v2/ML2_G_B_skip_1_normal
# outputs/skip_ablation_v2/ML2_G_B_skip_0_canny
# outputs/skip_ablation_v2/ML2_G_B_skip_0.5_c
# outputs/skip_ablation_v2/ML2_G_B_skip_1_canny
# -------------------------

# Best performance was two (still quite bad tough...)
# modified => ML2_G_B_skip_0.0_canny
# original => ML2_G_B_skip_0.5_canny => has better loss evolution than above 
# To train original ML2_G_B_skip_0.5_canny more time: 

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