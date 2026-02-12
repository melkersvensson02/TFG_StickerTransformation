#!/bin/bash
#SBATCH --job-name=cyle_turbo
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
ls -l src/Train_pix2pix_turbo.py


echo "We are about to run the python code!"
BASE_MODEL="/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo"
DATASET="/data/upftfg19/mfsvensson/Data_TFG/myUnPairedDataset"
RES=512
BS=4
OUTDIR="/home/mfsvensson/TFG_reps/Old-img2img-turbo/outputs/cycle_train_longer_diffSkip"
MAX_STEPS=8000

for SKIP_WEIGHT in 0.0 0.5 1.0; do
    CURRENT_OUTDIR="${OUTDIR}/cycle_train_longer_diffSkip_${SKIP_WEIGHT}"

    if [[ "${SKIP_WEIGHT}" == "0.0" ]]; then
        SKIP_ARGS="--vae_ignore_skip True --vae_skip_weight 1.0"
    else
        SKIP_ARGS="--vae_ignore_skip False --vae_skip_weight ${SKIP_WEIGHT}"
    fi

    echo "Running with SKIP_WEIGHT=${SKIP_WEIGHT}, output to ${CURRENT_OUTDIR}"

    accelerate launch src/Cyclegan_Train_turbo.py \
        --pretrained_model_name_or_path "${BASE_MODEL}" \
        --output_dir "${CURRENT_OUTDIR}" \
        --dataset_folder "${DATASET}" \
        --train_batch_size "${BS}" \
        --max_train_steps "${MAX_STEPS}" \
        --enable_xformers_memory_efficient_attention \
        --learning_rate="1e-5" --gradient_accumulation_steps=1 \
        --train_img_prep "resize_286_randomcrop_256x256_hflip" --val_img_prep "resize_256" \
        --viz_freq 25 \
        ${SKIP_ARGS}
done
echo "All training runs completed."