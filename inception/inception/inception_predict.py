# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A library to evaluate Inception on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import os.path
import time

import numpy as np
import tensorflow as tf

from inception.inception import image_processing
from inception.inception import inception_model as inception
from inception.inception.cv_data import ShopeeData

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('test_file_path', '',
                           """JPEG file to predict with graph""")
tf.app.flags.DEFINE_string('checkpoint_dir_image',
                           os.path.join('models',
                                        'image',
                                        '299x299x3',
                                        'inception',
                                        '100000'),
                           """Directory where to read model checkpoints.""")

# Flags governing the data used for prediction.
tf.app.flags.DEFINE_string('subset', 'validation',
                           """Either 'validation' or 'train'.""")

dataset = ShopeeData(subset=FLAGS.subset)


def predict(image_path, checkpoint_path=None):
    """Predict input file and returns logits"""
    with tf.Graph().as_default():
        # Convert image into tensor and do some pre-processing
        image_data = tf.read_file(image_path)
        image = image_processing.image_preprocessing(image_data, [], False)

        # reshape into correct dimensions for graph
        image = tf.cast(image, tf.float32)
        height = FLAGS.image_size
        width = FLAGS.image_size
        depth = 3
        image = tf.reshape(image, shape=[-1, height, width, depth])

        # Number of classes in the Dataset label set plus 1.
        # Label 0 is reserved for an (unused) background class.
        num_classes = dataset.num_classes() + 1

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits_op, _ = inception.inference(image, num_classes)

        probs_op = tf.nn.softmax(logits_op)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            inception.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            # restore model from checkpoint file
            if checkpoint_path is None:
                checkpoint_path = FLAGS.checkpoint_dir_image

            # assert tf.gfile.Exists(checkpoint_path)
            # saver.restore(sess, checkpoint_path)
            # print('%s: Successfully loaded model from %s' %
            #       (datetime.now(), checkpoint_path))

            assert tf.gfile.Exists(checkpoint_path)
            ckpt = tf.train.get_checkpoint_state(checkpoint_path)
            # print(ckpt.model_checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                if os.path.isabs(ckpt.model_checkpoint_path):
                    # Restores from checkpoint with absolute path.
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    # Restores from checkpoint with relative path.
                    saver.restore(sess, os.path.join(checkpoint_path, ckpt.model_checkpoint_path))

                # saver.restore(sess, checkpoint_path)
                print('%s: Successfully loaded model from %s' %
                      (datetime.now(), ckpt.model_checkpoint_path))
            else:
                print("No checkpoint file found")
                return None

            logits, probs = sess.run([logits_op, probs_op])

            return logits, probs


def load_graph(sess, checkpoint_path=None):
    """Predict logits of classes with input file"""
    #######################
    # Graph Definition
    image_path = tf.placeholder(tf.string, name='image_path_input')
    # Convert image into tensor and do some pre-processing
    image_data = tf.read_file(image_path)
    image = image_processing.image_preprocessing(image_data, [], False)

    # reshape into correct dimensions for graph
    image = tf.cast(image, tf.float32)
    height = FLAGS.image_size
    width = FLAGS.image_size
    depth = 3
    image = tf.reshape(image, shape=[-1, height, width, depth])

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    num_classes = dataset.num_classes() + 1

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits_op, _ = inception.inference(image, num_classes)

    probs_op = tf.nn.softmax(logits_op)

    ##########################################
    # Saver object definition
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        inception.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    #############################################
    # Restoring checkpoint files
    # restore model from checkpoint file
    if checkpoint_path is None:
        checkpoint_path = FLAGS.checkpoint_dir_image

    # assert tf.gfile.Exists(checkpoint_path)
    # saver.restore(sess, checkpoint_path)
    # print('%s: Successfully loaded model from %s' %
    #       (datetime.now(), checkpoint_path))

    assert tf.gfile.Exists(checkpoint_path)
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    # print(ckpt.model_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        if os.path.isabs(ckpt.model_checkpoint_path):
            # Restores from checkpoint with absolute path.
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            # Restores from checkpoint with relative path.
            saver.restore(sess, os.path.join(checkpoint_path, ckpt.model_checkpoint_path))

        # saver.restore(sess, checkpoint_path)
        print('%s: Successfully loaded model from %s' %
              (datetime.now(), ckpt.model_checkpoint_path))
    else:
        print("No checkpoint file found")
        return None, None

    return logits_op, probs_op


def main(unused_argv):
    # dataset = ShopeeData(subset=FLAGS.subset)
    logits, probs = predict(FLAGS.test_file_path, dataset)
    print(logits)
    print(probs)


if __name__ == '__main__':
    tf.app.run()
