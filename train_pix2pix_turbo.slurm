#!/bin/bash
#SBATCH --job-name=pix2pix_turbo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
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
accelerate launch src/Train_pix2pix_turbo.py \
    --pretrained_model_name_or_path="/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo" \
    --output_dir="outputs/myPairedDataset_ML2_L_B" \
    --dataset_folder="/data/upftfg19/mfsvensson/Data_TFG/myPairedDatasets" \
    --resolution=512 \
    --train_batch_size 4\
    --num_training_epochs 5 --data_set_type "modified" \
    --enable_xformers_memory_efficient_attention --viz_freq 25 \
    --track_val_fid \
    --mixed_precision="bf16" \
    --num_samples_eval 100 \
    --checkpointing_steps 1000\
    --lambda_clipsim 0.0 \
    --lambda_lpips 1.0 \
    --lambda_gan 0.0 \
    --lambda_l2 1.0 \
