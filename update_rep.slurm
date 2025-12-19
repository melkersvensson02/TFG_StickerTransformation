#!/bin/bash
#SBATCH --job-name=sync_code
#SBATCH --output=sync_code.out
#SBATCH --time=00:10:00

rsync -av --progress \
  --include="*/" \
  --include="*.py" \
  --exclude="*" \
  /home/melke/Documents/TFG_StickerTransformation/ \
  mfsvensson@cluster:/path/to/remote/project/
