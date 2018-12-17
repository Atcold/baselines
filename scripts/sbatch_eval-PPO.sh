# Allows named arguments
set -k

for model in PPO_0.2_lane-models/seed*; do
    sbatch submit_eval-PPO.slurm model=$model
done
