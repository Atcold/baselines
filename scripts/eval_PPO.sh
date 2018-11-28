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

srun python -m baselines.run \
    --alg=ppo2 \
    --env=I-80-v1 \
    --n_cond=20 \
    --seed=0 \
    --num_timesteps=0 \
    --eval \
    --gamma=0.99 \
    --load_path=$1
    #2>/dev/null
