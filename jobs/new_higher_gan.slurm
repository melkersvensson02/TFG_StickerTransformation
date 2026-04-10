#!/bin/bash
#SBATCH --job-name=new_higher_gan
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
OUTDIR="/home/mfsvensson/TFG_reps/Old-img2img-turbo/outputs/new_higher_gan"

# Higher GAN weight to counteract the strong cycle loss pulling towards identity.
# lambda_cycle_lpips=10 gives a very strong identity pull; lambda_gan=2 creates
# a better balance between "preserve structure" and "look like domain B".
# Note: ablation_1_higher_gan.slurm also tested lambda_gan=2 but with old code
# (CX gradient bug, no shared VGG, old validation metrics). This re-runs it cleanly.

accelerate launch src/cyclegan_train_turbo.py \
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
    --lambda_gan 2.0 \
    --lambda_cycle 1.0 \
    --lambda_cycle_lpips 10.0 \
    --lambda_idt 1.0 \
    --lambda_idt_lpips 1.0 \
    --lambda_gdl 0.0 \
    --lambda_cx 0.0 \


echo "Training completed."
