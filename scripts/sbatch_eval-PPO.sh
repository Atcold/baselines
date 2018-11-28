# Allows named arguments
set -k

for model in PPO-models/seed2*; do
    sbatch submit_eval-PPO.slurm model=$model
done
