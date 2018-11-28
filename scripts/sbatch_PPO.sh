# Allows named arguments
set -k

for seed in 0 1 2; do
    for lr in 3e-5 1e-4; do
        for gamma in 0.99; do
            sbatch submit_PPO.slurm \
                seed=$seed \
                lr=$lr \
                gamma=$gamma
        done
    done
done
