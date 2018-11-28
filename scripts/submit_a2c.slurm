#!/bin/bash
#
#SBATCH --job-name=A2C-i80
#SBATCH --output=logs/A2C-i80-%j.out
#SBATCH --error=logs/A2C-i80-%j.err
#SBATCH --time=48:00:00
#SBATCH --gres gpu:1
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=canziani@nyu.edu
#SBATCH --exclude="rose[1-9]"

echo "Running on $(hostname)"

source activate OpenAI
module load mpi/openmpi-x86_64
module load cuda-9.0

: <<'!'
# Export saving dir
date=$(date '+%Y-%m-%d_%H:%M:%S')
path="$HOME/atcold/RL-results"
exp="${date}_seed=${seed}_lr=${lr}_gamma=${gamma}"
export OPENAI_LOGDIR="$path/$exp"
!
seed=0
lr=7e-4
gamma=0.99

echo \
srun python -m baselines.run \
    --alg=a2c \
    --env=I-80-v1 \
    --n_cond=20 \
    --seed=$seed \
    --gamma=$gamma \
    --lr=$lr
python -m baselines.run \
    --alg=a2c \
    --env=I-80-v1 \
    --n_cond=20 \
    --seed=$seed \
    --gamma=$gamma \
    --lr=$lr
