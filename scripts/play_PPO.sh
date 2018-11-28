source activate OpenAI
module load mpi/openmpi-x86_64
module load cuda-9.0

python -m baselines.run \
    --alg=ppo2 \
    --env=I-80-v1 \
    --n_cond=20 \
    --seed=0 \
    --num_timesteps=0 \
    --play \
    --gamma=0.99 \
    --load_path=$1
    #2>/dev/null
