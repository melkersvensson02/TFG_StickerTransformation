#!/bin/bash
#SBATCH --job-name=pix2pix_turbo
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
s -l src/Train_pix2pix_turbo.py


echo "We are about to run the python code!"
# -------------------------
BASE_MODEL="/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo"
DATASET="/data/upftfg19/mfsvensson/Data_TFG/PairedDatasetSemantic"

RES=512
BS=4
MAX_STEPS=1102

# Paper baseline lambdas
LAMBDA_L2=1.0
LAMBDA_GAN=0.5
LAMBDA_LPIPS=5.0
LAMBDA_CLIPSIM=5.0

# Canny defaults
CANNY_LOW=100
CANNY_HIGH=200

ROOT_OUT="/home/mfsvensson/TFG_reps/Old-img2img-turbo/outputs/CLIP_ablation/"

# -------------------------
# A) BASELINE (no canny; no skip)
# -------------------------
OUTNAME="myPairedDatasetSemantic_baseline_noskip_nocanny"
OUTDIR="${ROOT_OUT}/${OUTNAME}"

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
  --checkpointing_steps 1000 \
  --lambda_l2 "${LAMBDA_L2}" \
  --lambda_gan "${LAMBDA_GAN}" \
  --lambda_lpips "${LAMBDA_LPIPS}" \
  --lambda_clipsim "${LAMBDA_CLIPSIM}" \
  --ignore_skip True

# -------------------------
# B) BASELINE + SKIP (0.5 / 1.0), CANNY ONLY
# -------------------------
for SKIP in 0.5 1.0; do
  OUTNAME="myPairedDatasetSemantic_baseline_skip${SKIP}_canny"
  OUTDIR="${ROOT_OUT}/${OUTNAME}"

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
    --checkpointing_steps 1000 \
    --lambda_l2 "${LAMBDA_L2}" \
    --lambda_gan "${LAMBDA_GAN}" \
    --lambda_lpips "${LAMBDA_LPIPS}" \
    --lambda_clipsim "${LAMBDA_CLIPSIM}" \
    --skip_weight "${SKIP}" \
    --use_canny_conditioning True \
    --canny_low_threshold "${CANNY_LOW}" \
    --canny_high_threshold "${CANNY_HIGH}"
done


RES=512
BS=4
MAX_STEPS=1102

# ML2_G setup (your request)
LAMBDA_L2=1.0
LAMBDA_GAN=1.0
LAMBDA_LPIPS=0.0   # keep off unless you explicitly want it

# Canny defaults
CANNY_LOW=100
CANNY_HIGH=200

ROOT_OUT="/home/mfsvensson/TFG_reps/Old-img2img-turbo/outputs/CLIP_ablation_v2/"

for CLIPW in 1.0 0.5 3.0; do
  for SKIP01 in 0 1; do

    OUTNAME="myPairedDatasetSemantic_ML2_G_CLIP_${CLIPW}_skip_${SKIP01}_canny"
    OUTDIR="${ROOT_OUT}/${OUTNAME}"

    SKIP_ARGS=""
    if [[ "${SKIP01}" == "0" ]]; then
      SKIP_ARGS="--ignore_skip True"
    else
      # “1” = keep skip enabled at default strength (no ignore flag needed)
      SKIP_ARGS=""
    fi

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
      --checkpointing_steps 1000 \
      --lambda_l2 "${LAMBDA_L2}" \
      --lambda_gan "${LAMBDA_GAN}" \
      --lambda_lpips "${LAMBDA_LPIPS}" \
      --lambda_clipsim "${CLIPW}" \
      --use_canny_conditioning True \
      --canny_low_threshold "${CANNY_LOW}" \
      --canny_high_threshold "${CANNY_HIGH}" \
      ${SKIP_ARGS}

  done
done


