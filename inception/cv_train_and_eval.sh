#!/usr/bin/env bash
# Script to train Google's Inception v3 on cv image data
# This script assumes data has been sharded into TFRecord files
#
# NOTE: when changing data sizes, remember to change the values of
#   num_examples_per_epoch
# in cv_data.py
#
# usage:
#   ./cv_train_and_eval.sh [new]

# Directories
# current directory
PROJECT_DIR="/home/wooppy/Projects/WooppyProjectC"
# Directory containing TFRecord data
DATA_DIR="${PROJECT_DIR}/training_data/image_files/organized/100000_excl_04"
# Directory to store trained checkpoint and event files
TRAIN_DIR="${PROJECT_DIR}/inception/cv_train/100000_excl_04"
# Directory to store evaluation files
EVAL_DIR="${PROJECT_DIR}/inception/cv_eval/100000_excl_04"

# Directory for pretrained Inception v3 models
INCEPTION_MODEL_DIR="${HOME}/inception-v3-model"
MODEL_PATH="${INCEPTION_MODEL_DIR}/model.ckpt-157585"

# Training and evaluation parameters
NUM_GPU=1
BATCH_SIZE=50
# MEM_FACTOR: input queue size, larger allows more shuffling across all data
# effects amount of CPU memory used. Multiples of ~4GB
MEM_FACTOR=4
# Number of test data to validate on
NUM_VALID_EXAMPLES=17013
# Number of training steps per evaluation
# Also how often checkpoint files are written
VALID_RATE=1000
# Total steps to train, multiple of VALID_RATE
TOTAL_STEPS=75000

bazel build //inception:cv_train
bazel build //inception:cv_eval

if [ "$1" == "new" ]; then
    echo "Usage: new transfer learning"
fi

a=0
while [ "$a" -lt "$TOTAL_STEPS" ]; do
    if test "$1" = "new" && test "$a" -eq 0; then
        echo "Starting transfer learning script"
        bazel-bin/inception/cv_train \
	        --num_gpu="${NUM_GPU}" \
	        --batch_size="${BATCH_SIZE}" \
	        --train_dir="${TRAIN_DIR}" \
	        --data_dir="${DATA_DIR}" \
	        --pretrained_model_checkpoint_path="${MODEL_PATH}" \
	        --fine_tune=True \
	        --initial_learning_rate=0.001 \
	        --input_queue_memory_factor="${MEM_FACTOR}" \
	        --max_steps="${VALID_RATE}"
    else
        echo "Continuing learning from prev. checkpoint"
        bazel-bin/inception/cv_train \
            --num_gpu="${NUM_GPU}" \
            --batch_size="${BATCH_SIZE}" \
            --train_dir="${TRAIN_DIR}" \
            --data_dir="${DATA_DIR}" \
            --checkpoint_dir="${TRAIN_DIR}" \
            --initial_learning_rate=0.001 \
            --input_queue_memory_factor="${MEM_FACTOR}" \
            --max_steps="${VALID_RATE}"
    fi

    echo "Starting Validation script"
    bazel-bin/inception/cv_eval \
        --eval_dir="${EVAL_DIR}" \
        --data_dir="${DATA_DIR}" \
        --subset=validation \
        --num_examples="${NUM_VALID_EXAMPLES}" \
        --checkpoint_dir="${TRAIN_DIR}" \
        --input_queue_memory_factor="${MEM_FACTOR}" \
        --run_once
    a=$(( a+${VALID_RATE} ))
    # Portable to all, but slower
    # a='expr "$a" + "$VALID_RATE"'
    # Mostly Portable except to Bourne shell and older Almquist
    # a=$(($a+$VALID_RATE))
done