#!/bin/sh

# Private variables
SCRIPT_DIR=$(dirname $(dirname $(realpath "$0")))/"scripts"

# Environment Setting
set -x  # print the commands

export CUDA_VISIBLE_DEVICES=0

# Arguments
NUM_GPUS=1
NUM_ENV_RUNNERS=0
NUM_LEARNERS=0
CHECKPOINT=""
LOG_DIR="./ray_results"
SEED=0
LR=1e-3
MAX_ITERS=100
CHECKPOINT_FREQ=10
BATCH_SIZE=512
MINIBATCH_SIZE=512
PY_ARGS=${@}  # Additional args

# Run the script
python ${SCRIPT_DIR}/"train.py" \
    --num_gpus ${NUM_GPUS} \
    --num_env_runners ${NUM_ENV_RUNNERS} \
    --num_learners ${NUM_LEARNERS} \
    --checkpoint "${CHECKPOINT}" \
    --log_dir "${LOG_DIR}" \
    --seed ${SEED} \
    --lr ${LR} \
    --max_iters ${MAX_ITERS} \
    --checkpoint_freq ${CHECKPOINT_FREQ} \
    --batch_size ${BATCH_SIZE} \
    --minibatch_size ${MINIBATCH_SIZE} \
    ${PY_ARGS}
