#!/usr/bin/env bash
# Script preprocesses raw image data and converts it into TFRecord
#
# The final output of this script are sharded TFRecord files containing
# serialized Example protocol buffers. See build_image_data.py for
# details of how the Example protocol buffer contains image data.
#
# Ensure the data is organized as follows:
#
#   $TRAIN_DIR/dog/image0.jpeg
#   $TRAIN_DIR/dog/image1.jpg
#    ...
#   $TRAIN_DIR/cat/weird-image.jpeg
#   $TRAIN_DIR/cat/my-image.jpeg
#    ...
#   $VALIDATION_DIR/dog/imageA.jpeg
#   $VALIDATION_DIR/dog/imageC.png
#    ...
#   $VALIDATION_DIR/cat/weird-image.PNG
#   $VALIDATION_DIR/cat/cat.JPG
#    ...
#
# The final output of this script appears as such:
#
#   $DATA_DIR/train-00000-of-01024
#   $DATA_DIR/train-00001-of-01024
#    ...
#   $DATA_DIR/train-01023-of-01024
#
# and
#
#   $DATA_DIR/validation-00000-of-00128
#   $DATA_DIR/validation-00001-of-00128
#    ...
#   $DATA_DIR/validation-00127-of-00128
#
# usage:
#   ./shard_data_to_tfrecord.sh

# Directories
# current directory
#PROJECT_DIR="/home/wooppy/Projects/WooppyProjectC"
PROJECT_DIR="/home/wooppy/Projects/WooppyProjectC/WooppyProjectC"
# Directory to write TFRecord to
DATA_DIR="${PROJECT_DIR}/training_data/inception_tfrecords"
# Directories containing raw train and validation images
#TRAIN_DATA_DIR="${DATA_DIR}/train"
#VALIDATION_DATA_DIR="${DATA_DIR}/validation"
TRAIN_DATA_DIR="${PROJECT_DIR}/training_data/converted_folder"
VALIDATION_DATA_DIR="${PROJECT_DIR}/training_data/converted_folder"
# File containing list of labels
LABELS_FILE="${PROJECT_DIR}/training_data/labels.txt"

bazel build //inception:build_image_data
# value of train_shards & validation_shards chosen so each TFRecord has
# around 1024 samples
# num_threads has to be a factor of both train_shards &
bazel-bin/inception/build_image_data \
	--train_directory="${TRAIN_DATA_DIR}" \
	--validation_directory="${VALIDATION_DATA_DIR}" \
	--output_directory="${DATA_DIR}" \
	--labels_file="${LABELS_FILE}" \
	--train_shards=1 \
	--validation_shards=1 \
	--num_threads=1