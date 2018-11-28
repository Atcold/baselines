#!/bin/bash
#
#SBATCH --job-name=PPO-i80
#SBATCH --output=logs/PPO-i80-%j.out
#SBATCH --error=logs/PPO-i80-%j.err
#SBATCH --time=48:00:00
#SBATCH --gres gpu:1
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=canziani@nyu.edu
#SBATCH --exclude="rose[1-9]"
#SBATCH --constraint="gpu_12gb&pascal"
#SBATCH -c 6  # 6 cores

echo "Running on $(hostname)"

source activate OpenAI
module load mpi/openmpi-x86_64
module load cuda-9.0

# Export saving dir
date=$(date '+%Y-%m-%d_%H:%M:%S')
path="$HOME/atcold/RL-results"
exp="${date}_seed=${seed}_lr=${lr}_gamma=${gamma}"
export OPENAI_LOGDIR="$path/$exp"

: <<'!'
seed=0
lr=3e-4
gamma=0.99
!

echo \
srun python -m baselines.run \
    --alg=ppo2 \
    --env=I-80-v1 \
    --n_cond=20 \
    --seed=$seed \
    --gamma=$gamma \
    --lr=$lr \
    --nminibatches=6 \
    --num_env=6 \
    --num_timesteps=1e7 \
    --load_path=load-PPO-train-0.2_lane-2d/seed${seed}_lr$lr

# cd $HOME/baselines/
# srun python run.py \
srun python -m baselines.run \
    --alg=ppo2 \
    --env=I-80-v1 \
    --n_cond=20 \
    --seed=$seed \
    --gamma=$gamma \
    --lr=$lr \
    --nminibatches=6 \
    --num_env=6 \
    --num_timesteps=1e7 \
    --load_path=load-PPO-train-0.2_lane-2d/seed${seed}_lr$lr
