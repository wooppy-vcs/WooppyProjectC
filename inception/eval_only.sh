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
#PROJECT_DIR="/home/wooppy/Projects/WooppyProjectC"
PROJECT_DIR="/home/wooppy/Projects/WooppyProjectC/WooppyProjectC"
# Directory containing TFRecord data
#DATA_DIR="${PROJECT_DIR}/training_data/image_files"
DATA_DIR="${PROJECT_DIR}/training_data/inception_tfrecords"
# Directory to store trained checkpoint and event files
#TRAIN_DIR="${PROJECT_DIR}/WooppyProjectC/inception/cv_train/trial"
TRAIN_DIR="${PROJECT_DIR}/inception/cv_train/multilabel"
# Directory to store evaluation files
#EVAL_DIR="${PROJECT_DIR}/WooppyProjectC/inception/cv_eval/trial"
EVAL_DIR="${PROJECT_DIR}/inception/cv_eval/multilabel"

# Directory for pretrained Inception v3 models
INCEPTION_MODEL_DIR="${HOME}/inception-v3-model"
MODEL_PATH="${INCEPTION_MODEL_DIR}/model.ckpt-157585"

# Training and evaluation parameters
NUM_GPU=1
BATCH_SIZE=1
# MEM_FACTOR: input queue size, larger allows more shuffling across all data
# effects amount of CPU memory used. Multiples of ~4GB
MEM_FACTOR=8
# Number of test data to validate on
NUM_VALID_EXAMPLES=11
# Number of training steps per evaluation
# Also how often checkpoint files are written
VALID_RATE=10000
# Total steps to train, multiple of VALID_RATE
TOTAL_STEPS=10000

bazel build //inception:cv_train
bazel build //inception:cv_eval

if [ "$1" == "new" ]; then
    echo "Usage: new transfer learning"
fi

a=0
#	        --pretrained_model_checkpoint_path="${MODEL_PATH}" \
while [ "$a" -lt "$TOTAL_STEPS" ]; do
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