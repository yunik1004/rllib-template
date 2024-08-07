#!/bin/sh

# Private variables
SCRIPT_DIR=$(dirname $(dirname $(realpath "$0")))/"scripts"

# Environment Setting
set -x  # print the commands

export CUDA_VISIBLE_DEVICES=0

# Arguments
ACCELERATOR="gpu"
CHECKPOINT=""
OUTPUT_DIR="./ray_results"
SEED=0
MAX_STEPS=-1
PY_ARGS=${@}  # Additional args

# Run the script
python ${SCRIPT_DIR}/"inference.py" \
    --accelerator "${ACCELERATOR}" \
    --checkpoint "${CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed ${SEED} \
    --max_steps ${MAX_STEPS} \
    ${PY_ARGS}
