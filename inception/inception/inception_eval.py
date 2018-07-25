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

from inception import image_processing
from inception import inception_model as inception

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/imagenet_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/imagenet_train',
                           """Directory where to read model checkpoints.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_integer('num_examples', 11,
                            """Number of examples to run. Note that the eval """
                            """ImageNet dataset contains 50000 examples.""")
tf.app.flags.DEFINE_string('subset', 'validation',
                           """Either 'validation' or 'train'.""")
tf.app.flags.DEFINE_integer('num_classes', 13,
                            """Number of classes""")


# def _eval_once(saver, summary_writer, top_1_op, top_5_op, summary_op):
#     """Runs Eval once.
#
#   Args:
#     saver: Saver.
#     summary_writer: Summary writer.
#     top_1_op: Top 1 op.
#     top_5_op: Top 5 op.
#     summary_op: Summary op.
#   """
#     with tf.Session() as sess:
#         ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
#         if ckpt and ckpt.model_checkpoint_path:
#             if os.path.isabs(ckpt.model_checkpoint_path):
#                 # Restores from checkpoint with absolute path.
#                 saver.restore(sess, ckpt.model_checkpoint_path)
#             else:
#                 # Restores from checkpoint with relative path.
#                 saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
#                                                  ckpt.model_checkpoint_path))
#
#             # Assuming model_checkpoint_path looks something like:
#             #   /my-favorite-path/imagenet_train/model.ckpt-0,
#             # extract global_step from it.
#             global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#             print('Successfully loaded model from %s at step=%s.' %
#                   (ckpt.model_checkpoint_path, global_step))
#         else:
#             print('No checkpoint file found')
#             return
#
#         # Start the queue runners.
#         coord = tf.train.Coordinator()
#         try:
#             threads = []
#             for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
#                 threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
#                                                  start=True))
#
#             num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
#             # Counts the number of correct predictions.
#             count_top_1 = 0.0
#             count_top_5 = 0.0
#             total_sample_count = num_iter * FLAGS.batch_size
#             step = 0
#
#             print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
#             start_time = time.time()
#             while step < num_iter and not coord.should_stop():
#                 top_1, top_5 = sess.run([top_1_op, top_5_op])
#                 count_top_1 += np.sum(top_1)
#                 count_top_5 += np.sum(top_5)
#                 step += 1
#                 if step % 20 == 0:
#                     duration = time.time() - start_time
#                     sec_per_batch = duration / 20.0
#                     examples_per_sec = FLAGS.batch_size / sec_per_batch
#                     print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
#                           'sec/batch)' % (datetime.now(), step, num_iter,
#                                           examples_per_sec, sec_per_batch))
#                     start_time = time.time()
#
#             # Compute precision @ 1.
#             precision_at_1 = count_top_1 / total_sample_count
#             recall_at_5 = count_top_5 / total_sample_count
#             print('%s: precision @ 1 = %.4f recall @ 5 = %.4f [%d examples]' %
#                   (datetime.now(), precision_at_1, recall_at_5, total_sample_count))
#
#             summary = tf.Summary()
#             summary.ParseFromString(sess.run(summary_op))
#             summary.value.add(tag='Precision @ 1', simple_value=precision_at_1)
#             summary.value.add(tag='Recall @ 5', simple_value=recall_at_5)
#             summary_writer.add_summary(summary, global_step)
#
#         except Exception as e:  # pylint: disable=broad-except
#             coord.request_stop(e)
#
#         coord.request_stop()
#         coord.join(threads, stop_grace_period_secs=10)
#
#
# def evaluate(dataset):
#     """Evaluate model on Dataset for a number of steps."""
#     with tf.Graph().as_default():
#         # Get images and labels from the dataset.
#         images, labels = image_processing.inputs(dataset)
#
#         # Number of classes in the Dataset label set plus 1.
#         # Label 0 is reserved for an (unused) background class.
#         num_classes = dataset.num_classes() + 1
#
#         # Build a Graph that computes the logits predictions from the
#         # inference model.
#         logits, _ = inception.inference(images, num_classes)
#
#         # Calculate predictions.
#         top_1_op = tf.nn.in_top_k(logits, labels, 1)
#         top_5_op = tf.nn.in_top_k(logits, labels, 5)
#
#         # Restore the moving average version of the learned variables for eval.
#         variable_averages = tf.train.ExponentialMovingAverage(
#             inception.MOVING_AVERAGE_DECAY)
#         variables_to_restore = variable_averages.variables_to_restore()
#         saver = tf.train.Saver(variables_to_restore)
#
#         # Build the summary operation based on the TF collection of Summaries.
#         summary_op = tf.summary.merge_all()
#
#         graph_def = tf.get_default_graph().as_graph_def()
#         summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,
#                                                graph_def=graph_def)
#
#         while True:
#             _eval_once(saver, summary_writer, top_1_op, top_5_op, summary_op)
#             if FLAGS.run_once:
#                 break
#             time.sleep(FLAGS.eval_interval_secs)


def _eval_once(saver, summary_writer, label_bool_op, summary_op):
    """Runs Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_1_op: Top 1 op.
    top_5_op: Top 5 op.
    summary_op: Summary op.
  """
    with tf.Session() as sess:
        print(FLAGS.checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            if os.path.isabs(ckpt.model_checkpoint_path):
                # Restores from checkpoint with absolute path.
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                # Restores from checkpoint with relative path.
                saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
                                                 ckpt.model_checkpoint_path))

            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/imagenet_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('Successfully loaded model from %s at step=%s.' %
                  (ckpt.model_checkpoint_path, global_step))
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            # Counts the number of correct predictions.
            # count_top_1 = 0.0
            # count_top_5 = 0.0
            truth_count = 0.0

            total_sample_count = num_iter * FLAGS.batch_size
            total_sample_class_count = total_sample_count * FLAGS.num_classes
            step = 0

            print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
            start_time = time.time()
            while step < num_iter and not coord.should_stop():

                # top_1, top_5 = sess.run([top_1_op, top_5_op])
                # count_top_1 += np.sum(top_1)
                # count_top_5 += np.sum(top_5)
                # print(FLAGS.batch_size)
                # print(FLAGS.num_examples)
                # print(num_iter)
                # print(step)

                # label_bools_per_batch = sess.run([label_bool_op])

                label_bool_op_output = sess.run([label_bool_op])
                # print(label_bool_op_output)
                label_bools_per_batch = label_bool_op_output[0][0]
                sigmoid_logits_per_batch = label_bool_op_output[0][1]
                labels_per_batch = label_bool_op_output[0][2]

                truth_count += np.sum(label_bools_per_batch)
                # print(labels_per_batch)
                # print(sigmoid_logits_per_batch)
                # print(label_bools_per_batch)
                sx = ""
                lol = sigmoid_logits_per_batch[0]
                for x in lol:
                    sx += str(x) + "\t"
                print(sx)

                step += 1
                if step % 20 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / 20.0
                    examples_per_sec = FLAGS.batch_size / sec_per_batch
                    print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                          'sec/batch)' % (datetime.now(), step, num_iter,
                                          examples_per_sec, sec_per_batch))
                    start_time = time.time()

            # Compute precision @ 1.
            # precision_at_1 = count_top_1 / total_sample_count
            # recall_at_5 = count_top_5 / total_sample_count
            overall_accuracy = truth_count / total_sample_class_count
            print('Truth count = %d' % truth_count)
            # print('%s: precision @ 1 = %.4f recall @ 5 = %.4f [%d examples]' %
            #       (datetime.now(), precision_at_1, recall_at_5, total_sample_count))

            print('%s: overall_accuracy = %.4f [%d examples classes]' %
                  (datetime.now(), overall_accuracy, total_sample_class_count))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            # summary.value.add(tag='Precision @ 1', simple_value=precision_at_1)
            # summary.value.add(tag='Recall @ 5', simple_value=recall_at_5)
            summary.value.add(tag='Overall Accuracy', simple_value=overall_accuracy)
            summary_writer.add_summary(summary, global_step)

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate(dataset):
    """Evaluate model on Dataset for a number of steps."""
    with tf.Graph().as_default():
        # Get images and labels from the dataset.
        images, labels = image_processing.inputs(dataset)

        # Number of classes in the Dataset label set plus 1.
        # Label 0 is reserved for an (unused) background class.
        num_classes = dataset.num_classes() + 1

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits, _ = inception.inference(images, num_classes)
        # print(logits.get_shape())
        # print(labels.get_shape())

        # Calculate predictions.
        # top_1_op = tf.nn.in_top_k(logits, labels, 1)
        # top_5_op = tf.nn.in_top_k(logits, labels, 5)
        label_bool_op = evaluate_multilabel(logits, labels)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            inception.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,
                                               graph_def=graph_def)

        # while True:
        #     _eval_once(saver, summary_writer, top_1_op, top_5_op, summary_op)
        #     if FLAGS.run_once:
        #         break
        #     time.sleep(FLAGS.eval_interval_secs)
        while True:
            _eval_once(saver, summary_writer, label_bool_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def evaluate_multilabel(logits, labels):
    # delta = tf.constant(value=0.125,
    #                     dtype=tf.float32,
    #                     shape=tf.shape(labels))
    sigmoid_logits = tf.nn.sigmoid(logits)
    delta = tf.constant(value=0.125,
                        dtype=tf.float32)
    # labels = tf.Print(labels, [labels])
    # delta = tf.Print(delta, [delta])
    delta_for_less_bool = tf.add(labels, delta)
    # delta_for_less_bool = tf.Print(delta_for_less_bool, [delta_for_less_bool])
    less_bool = tf.less_equal(sigmoid_logits, delta_for_less_bool)
    # print("inception_eval.py line 298")
    # print("shape of less_bool is: ")
    # print(less_bool.get_shape())

    delta_for_more_bool = tf.subtract(labels, delta)
    more_bool = tf.greater(sigmoid_logits, delta_for_more_bool)

    label_bool = tf.logical_and(less_bool, more_bool)

    return label_bool, sigmoid_logits, labels
    # return label_bool