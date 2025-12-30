#!/bin/bash
#SBATCH --job-name=sync_code
#SBATCH --output=sync_code.out
#SBATCH --time=00:10:00

rsync -av --progress -e "ssh -p 2122"   --include="*/" --include="*.slurm"  --include="*.py"   --exclude="*" ./ mfsvensson@pirineus3.csuc.cat:/home/mfsvensson/TFG_reps/Old-img2img-turbo/
