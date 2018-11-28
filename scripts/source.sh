function OpenAI-on {
    source activate OpenAI
    module load mpi/openmpi-x86_64
    module load cuda-9.0
}

function OpenAI-off {
    source deactivate
    module unload mpi/openmpi-x86_64
    module unload cuda-9.0
}
