# Allows named arguments
set -k

for seed in 0; do
    for lr in 2e-4 7e-4 2e-3; do
        for gamma in 0.99 0.95 0.90; do
            sbatch submit_a2c.slurm \
                seed=$seed \
                lr=$lr \
                gamma=$gamma
        done
    done
done
