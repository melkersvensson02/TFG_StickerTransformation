#!/bin/bash
#SBATCH --job-name=new_pix2pix_baseline
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

# Paired dataset — use whichever you compare the finetuned model against.
# Change to NewPairedDatasetSemantic + data_set_type=original if comparing
# against the Semantic finetune.
DATASET="/data/upftfg19/mfsvensson/Data_TFG/NewPairedDatasetNonSemantic"
OUTDIR="/home/mfsvensson/TFG_reps/Old-img2img-turbo/outputs/new_pix2pix_baseline"

# Pix2Pix trained directly from SD-Turbo base — NO CycleGAN pretraining.
# This is the fair baseline to compare against fineTune_cyclegan_a2b_notsemantic:
# same paired data, same steps, same hyperparams; only difference is the starting
# checkpoint (SD-Turbo base vs a CycleGAN-pretrained model).

accelerate launch src/train_pix2pix_turbo.py \
    --pretrained_model_name_or_path "/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo" \
    --output_dir "${OUTDIR}" \
    --dataset_folder="${DATASET}" \
    --resolution=512 \
    --train_batch_size=1 \
    --enable_xformers_memory_efficient_attention \
    --track_val_fid \
    --max_train_steps=6002 \
    --checkpointing_steps=1000 \
    --eval_freq=250 \
    --lambda_clipsim=5.0 \
    --lambda_l2=1.0 \
    --lambda_lpips=5.0 \
    --lambda_gan=0.5 \

echo "Training completed."
