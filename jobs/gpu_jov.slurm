#!/bin/bash
#SBATCH --job-name=cycle_turbo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/cycle.out
#SBATCH --error=logs/cycle.err

GPU_LOG=gpu_info.txt
echo "Logging GPU usage to $GPU_LOG"
cd /home/mfsvensson/TFG_reps/Old-img2img-turbo
echo "Running from PWD: $(pwd)"

module load conda
source activate Pined-img2img-turbo-fixed

echo "Running on host: $(hostname)"
echo "CUDA devices available:"
nvidia-smi

# Background GPU logger (every 5 seconds)
(
  while true; do
    echo "===== $(date) =====" >> $GPU_LOG
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv >> $GPU_LOG
    echo "" >> $GPU_LOG
    sleep 5
  done
) &
GPU_LOG_PID=$!

# Optional but VERY useful
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE_MODEL="/data/upftfg19/mfsvensson/TFG_weights/img2img-turbo"
DATASET="/data/upftfg19/mfsvensson/Data_TFG/Final_unpaired"
OUTDIR_BASE="/home/mfsvensson/TFG_reps/Old-img2img-turbo/outputs/GPU_tests"

RES=512
MAX_STEPS=1

# Grid
BATCH_SIZES=(1 2 4)
FID_SIZES=(25 75)

echo "Starting grid experiments..."

for BS in "${BATCH_SIZES[@]}"; do
  for FID in "${FID_SIZES[@]}"; do

    OUTDIR="${OUTDIR_BASE}/bs${BS}_fid${FID}"
    mkdir -p "${OUTDIR}"
    echo "=============================================="
    echo "Running experiment: BS=${BS}, FID=${FID}"
    echo "Output dir: ${OUTDIR}"
    echo "=============================================="

    accelerate launch src/GPU_cycle.py \
        --pretrained_model_name_or_path "${BASE_MODEL}" \
        --output_dir "${OUTDIR}" \
        --dataset_folder "${DATASET}" \
        --train_batch_size "${BS}" \
        --max_train_steps "${MAX_STEPS}" \
        --enable_xformers_memory_efficient_attention \
        --learning_rate 1e-5 \
        --gradient_accumulation_steps 1 \
        --train_img_prep "resize_512" \
        --val_img_prep "resize_512" \
        --vae_ignore_skip False \
        --vae_skip_weight 1.0 \
        --validation_num_images "${FID}" \
        >> logs/cycle.out 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "!!!!!!!!!!!!!! ERROR DETECTED !!!!!!!!!!!!!!" >> logs/cycle.out
        echo "Experiment BS=${BS}, FID=${FID} crashed." >> logs/cycle.out
        echo "Exit code: $EXIT_CODE" >> logs/cycle.out
        echo "Likely CUDA OOM or runtime error." >> logs/cycle.out
        echo "---------------------------------------------" >> logs/cycle.out
    else
        echo "Experiment BS=${BS}, FID=${FID} completed successfully." >> logs/cycle.out
    fi

  done
done

echo "All experiments finished."

# Kill GPU logger
kill $GPU_LOG_PID
