#!/bin/bash
#SBATCH --job-name=cycleGAN_turbo
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
ls -l src/Cyclegan_Train_turbo.py


echo "We are about to run the python code!"
export NCCL_P2P_DISABLE=1
accelerate launch src/Cyclegan_Train_turbo.py \
    --pretrained_model_name_or_path="/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo" \
    --output_dir="outputs/myZebraHorse_v2/" \
    --dataset_folder "/data/upftfg19/mfsvensson/Data_TFG/dataImg2ImgUnpairedFromRep/my_horse2zebra" \
    --train_img_prep "resize_286_randomcrop_256x256_hflip" --val_img_prep "no_resize" \
    --learning_rate="1e-5" --max_train_steps=25000 \
    --train_batch_size 4 --gradient_accumulation_steps=1 \
    --enable_xformers_memory_efficient_attention --validation_steps 250 \
    --lambda_gan 0.5 --lambda_idt 1 --lambda_cycle 1 --validation_num_images 100\
