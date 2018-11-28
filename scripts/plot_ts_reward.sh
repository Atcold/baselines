source activate OpenAI

python -m baselines.results_plotter \
    --task_name traffic \
    --dirs "$@"
