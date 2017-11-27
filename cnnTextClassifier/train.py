#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from cnnTextClassifier import data_helpers
# from cnnTextClassifier.data_helpers import calculate_weight
from cnnTextClassifier.text_cnn import TextCNN
from tensorflow.contrib import learn
import yaml

from cnnTextClassifier.text_cnn_v1 import TextCNNv1
from cnnTextClassifier.text_cnn_v2 import TextCNNv2

tf.flags.DEFINE_string("classifier_type", "-Scenario", "classifier type")
tf.flags.DEFINE_string("setting", "new-len90-CNNv1-featuresmap32_64-filtersize345-oneFC", "classifier setting")

# Parameters
# ==================================================
# tf.flags.DEFINE_string("checkpoint_dir", "runs/1499936835/checkpoints", "")
tf.flags.DEFINE_string("checkpoint_dir", "", "")
# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_integer("sentences_column", 0, "Column number of sentence data in data txt file")
tf.flags.DEFINE_integer("tags_column", 1, "Column number of tags in data txt file")


# Model Hyperparameters
# tf.flags.DEFINE_boolean("build_tags_vocab", True, "Enable building tags vocab if true")
tf.flags.DEFINE_boolean("enable_word_embeddings", True, "Enable/disable the word embedding (default: True)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
# tf.flags.DEFINE_integer("num_filters", 32, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("num_filters_layer1", 32, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("num_filters_layer2", 64, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("num_filters_layer3", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.5, "L2 regularization lambda (default: 0.0)")

# tf.flags.DEFINE_boolean("enable_word_embeddings", False, "Enable/disable the word embedding (default: True)")
# tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
# tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
# tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
# tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
# tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
# tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
# tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
# tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

dataset_name = cfg["datasets"]["default"]

if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
    embedding_name = cfg['word_embeddings']['default']
    embedding_dimension = cfg['word_embeddings'][embedding_name]['dimension']
    print("{} = {}".format("WORD_EMBEDDINGS", embedding_name))
else:
    embedding_dimension = FLAGS.embedding_dim
    print("{} = {}".format("WORD_EMBEDDINGS", "NONE"))

print("{} = {}".format("EMBEDDING DIMENSION", str(embedding_dimension)))
print("=======================================================")

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
datasets = None
print("dataset_name : " + dataset_name)

if dataset_name == "mrpolarity":
    datasets = data_helpers.get_datasets_mrpolarity(cfg["datasets"][dataset_name]["positive_data_file"]["path"],
                                                    cfg["datasets"][dataset_name]["negative_data_file"]["path"])
elif dataset_name == "20newsgroup":
    datasets = data_helpers.get_datasets_20newsgroup(subset="train",
                                                     categories=cfg["datasets"][dataset_name]["categories"],
                                                     shuffle=cfg["datasets"][dataset_name]["shuffle"],
                                                     random_state=cfg["datasets"][dataset_name]["random_state"])
elif dataset_name == "localdata":
    datasets = data_helpers.get_datasets_localdata(container_path=cfg["datasets"][dataset_name]["container_path"],
                                                     categories=cfg["datasets"][dataset_name]["categories"],
                                                     shuffle=cfg["datasets"][dataset_name]["shuffle"],
                                                     random_state=cfg["datasets"][dataset_name]["random_state"])
elif dataset_name == "localdatasingledata":
    datasets = data_helpers.get_datasets_localdatasinglefile(data_file=cfg["datasets"][dataset_name]["data_file"]["path"],
                                                             categories=cfg["datasets"][dataset_name]["categories"])

elif dataset_name == "localdatacategorizedbyfilename":
    datasets = data_helpers.get_datasets_localdatacategorizedbyfilename(container_path=cfg["datasets"][dataset_name]["data_file"]["path"],
                                                             categories_dict=cfg["datasets"][dataset_name]["categories_dict"])
elif dataset_name == "localfile":
    datasets=data_helpers.get_datasets(data_path=cfg["datasets"][dataset_name]["data_file"]["path"],
                                       vocab_tags_path=cfg["datasets"][dataset_name]["vocab_write_path"]["path"],
                                       class_weights_path=cfg["datasets"][dataset_name]["class_weights_path"]["path"],
                                       sentences=FLAGS.sentences_column, tags=FLAGS.tags_column)

x_text, y = data_helpers.load_data_labels(datasets)

# Build vocabulary
### max_document_length = max([len(x.split(" ")) for x in x_text])

# print("max_document_length: " + str(max_document_length))

max_document_length = 90
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
# print(y_shuffled.shape)
# ==================================================

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
# print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
# print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
# print(x_train)
# print(y_train)
print("=======================================================")
print("Data Details:")
print('Train/Dev split = %d/%d' % (len(y_train), len(y_dev)))
print('Train shape = ', x_train.shape)
print('Dev shape = ', x_dev.shape)
print('Vocab_size = ', len(vocab_processor.vocabulary_))
print('Sentence max words = ', max_document_length)
print("=======================================================")

weightsArray = datasets['class_weights']
print("Weights Array:")
print(weightsArray)

# Training

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNNv2(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=embedding_dimension,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters_layer1=FLAGS.num_filters_layer1,
            num_filters_layer2=FLAGS.num_filters_layer2,
            # num_filters_layer3=FLAGS.num_filters_layer3,
            l2_reg_lambda=FLAGS.l2_reg_lambda
            ,
            weights_array=weightsArray)

        # # restoring from the checkpoint file
        # ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        # tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))+FLAGS.classifier_type
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp + "-" + FLAGS.setting))
        print("=======================================================")
        print("Writing to {}\n".format(out_dir))
        print("=======================================================")

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
        weighted_acc_summary = tf.summary.scalar("weighted_accuracy", cnn.weighted_accuracy)
        weighted_prec_summary = tf.summary.scalar("weighted_precision", cnn.weighted_precision)
        weighted_f1_summary = tf.summary.scalar("weighted_f1", cnn.weighted_f1)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, weighted_acc_summary, weighted_prec_summary, weighted_f1_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary, weighted_acc_summary, weighted_prec_summary, weighted_f1_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
            vocabulary = vocab_processor.vocabulary_
            initW = None
            if embedding_name == 'word2vec':
                # load embedding vectors from the word2vec
                print("Load word2vec file {}".format(cfg['word_embeddings']['word2vec']['path']))
                initW = data_helpers.load_embedding_vectors_word2vec(vocabulary,
                                                                     cfg['word_embeddings']['word2vec']['path'],
                                                                     cfg['word_embeddings']['word2vec']['binary'])
                # print(len(initW))
                print("word2vec file has been loaded...")
                print("=======================================================")
            elif embedding_name == 'glove':
                # load embedding vectors from the glove
                print("Load glove file {}".format(cfg['word_embeddings']['glove']['path']))
                initW = data_helpers.load_embedding_vectors_glove(vocabulary,
                                                                  cfg['word_embeddings']['glove']['path'],
                                                                  embedding_dimension)
                print("glove file has been loaded\n")
                print("=======================================================")
            sess.run(cnn.W.assign(initW))

        if FLAGS.checkpoint_dir:
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            # print("fdsfdsfsdf")
            _, step, summaries, loss, accuracy, weighted_accuracy, weighted_f1, weighted_precision = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.weighted_accuracy, cnn.weighted_f1, cnn.weighted_precision],
                feed_dict)
            # print("fdsfdsfsdf")
            time_str = datetime.datetime.now().isoformat()
            print(
                "{}: step {}, loss {:g}, acc {:g}, wacc {:g}, wp {:g}, wf1 {:g}\n".format(time_str, step, loss, accuracy,
                                                                                         weighted_accuracy,
                                                                                         weighted_precision,
                                                                                         weighted_f1))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            # JIALER MOD COMMENTED
            # step, summaries, loss, accuracy, weighted_accuracy = sess.run(
            #     [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.weighted_accuracy],
            #     feed_dict)
            step, summaries, loss, accuracy, weighted_accuracy, weighted_f1, weighted_precision = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.weighted_accuracy, cnn.weighted_f1, cnn.weighted_precision],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print(
                "{}: step {}, loss {:g}, acc {:g}, wacc {:g}, wp {:g}, wf1 {:g}\n".format(time_str, step, loss, accuracy,
                                                                                         weighted_accuracy,
                                                                                         weighted_precision,
                                                                                         weighted_f1))
            if writer:
                writer.add_summary(summaries, step)
        # print(x_train)
        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        print("Training Starting.......")
        # Training loop. For each batch...
        i=0
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("=======================================================")
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("=======================================================")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
                print("Checkpoint Number = {}".format(i))
                print("=======================================================")
                i += 1

