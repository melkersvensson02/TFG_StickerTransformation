#!/bin/bash
#SBATCH --job-name=git_push
#SBATCH --output=git_push.out
#SBATCH --time=00:05:00

cd /home/melke/Documents/TFG_StickerTransformation
git add src/*.py *.py  # or adjust paths
git commit -m "Auto commit from Slurm"
git push origin main
