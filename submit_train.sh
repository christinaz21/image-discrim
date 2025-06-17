#!/bin/bash
#
# Job name:
#SBATCH --job-name=resnet50_train
#
# Output and error logs (you’ll find these under ./logs/):
#SBATCH --output=logs/resnet50_%j.out
#SBATCH --error=logs/resnet50_%j.err

# Request one node, one task, one GPU:
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#
# CPUs & memory per task:
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#
# Maximum runtime (D-HH:MM:SS):
#SBATCH --time=2:59:00


# ---------------------------------------
# 1) Set up your environment
# ---------------------------------------

# load modules or conda environments here
source ~/.bashrc
eval "$(conda shell.bash hook)"  # this is needed to load python packages correctly


# Activate your conda env (that has torch, torchvision, etc.)
conda activate images

# Create logs directory if needed
mkdir -p logs

# ---------------------------------------
# 2) Run training
# ---------------------------------------
srun python train.py \
    --data-dir /scratch/gpfs/cz5047/discrim \
    --epochs 50 \

# ---------------------------------------
# 3) (Optional) Post-processing
# ---------------------------------------
# e.g., copy checkpoints to permanent storage, send notification, etc.
