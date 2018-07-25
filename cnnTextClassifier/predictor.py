#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib import learn
import argparse
from cnnTextClassifier import data_helpers
# import yaml


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

def predict(x_raw, checkpoint_dir):
    # with open("config.yml", 'r') as ymlfile:
    #     cfg = yaml.load(ymlfile)

    # Parameters
    # ==================================================

    # Data Parameters

    # Eval Parameters
    # tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
    # if "checkpoint_dir" not in tf.flags:
    tf.app.flags.FLAGS = tf.flags._FlagValues()
    tf.flags._global_parser = argparse.ArgumentParser()
    tf.flags.DEFINE_string("checkpoint_dir", checkpoint_dir, "Checkpoint directory from training run")
    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    # print("\nParameters:")
    # for attr, value in sorted(FLAGS.__flags.items()):
    #     print("{}={}".format(attr.upper(), value))
    # print("")

    # Map data into vocabulary
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    print("this is x_raw")
    print(x_raw)
    x_raw = [data_helpers.clean_str(x_raw)]
    x_test = np.array(list(vocab_processor.transform(x_raw)))
    # print(x_test)
    # print("\nEvaluating...\n")

    # Evaluation
    # ==================================================
    # print(FLAGS.checkpoint_dir)
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            scores = graph.get_operation_by_name("output/scores").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Collect the predictions here

            predictions, scores = sess.run([predictions, scores],
                                           {input_x: x_test, dropout_keep_prob: 1.0})
            prediction = predictions[0]
            # probabilities = softmax(scores)[0]
            probabilities_op = tf.nn.sigmoid(scores)
            probabilities = sess.run(probabilities_op)

    # datasets = {"target_names": cfg["datasets"]["localdatasingledata"]["categories"]}
    # categories = datasets["target_names"]

    return prediction, probabilities