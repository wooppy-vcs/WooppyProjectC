#! /usr/bin/env python
import random

import numpy
import tensorflow as tf
import numpy as np
import os
import datetime
from cnnTextClassifier import data_helpers
from cnnTextClassifier.data_analyser import Counter
from cnnTextClassifier.text_cnn import TextCNN
from tensorflow.contrib import learn


def train(config, model=TextCNN):

    # Load data
    print("Loading data...")
    print("dataset_name : " + config.dataset_name)

    # if not config.enable_char:
    #     datasets=data_helpers.get_datasets(data_path=config.training_path, vocab_tags_path=config.tags_vocab_path,
    #                                        sentences=config.data_column, tags=config.tags_column)
    # else:

    datasets = data_helpers.get_datasets(data_path=config.training_path, vocab_tags_path=config.tags_vocab_path,
                                         vocab_char_path=config.char_vocab_path, config=config,
                                         sentences=config.data_column, tags=config.tags_column)
    # datasets_val = data_helpers.get_datasets(data_path=config.test_path, vocab_tags_path=config.tags_vocab_path,
    #                                          sentences=config.data_column, tags=config.tags_column)

    # Transforming labels to one-hot vectors and apply padding or truncate according to max_document_length
    x_text, y = data_helpers.load_data_labels(datasets)
    vocab_processor = learn.preprocessing.VocabularyProcessor(config.doc_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Char level processing
    if config.enable_char:
        processing_word = data_helpers.get_processing_word(vocab_chars=datasets['vocab_chars'])
        char_ids = []
        for sentence in x_text:
            words_raw = sentence.strip().split(" ")
            words = [processing_word(w) for w in words_raw]
            char_ids += [words]
        char_ids, word_lengths = data_helpers.pad_sequences(config.doc_length, char_ids, pad_tok=0, nlevels=2)
        char_ids = numpy.delete(char_ids, numpy.s_[config.doc_length:], 1)
        word_lengths = numpy.delete(word_lengths, numpy.s_[config.doc_length:], 1)

    if config.enable_char:
        data = list(zip(x, y, char_ids, word_lengths))
    else:
        data = list(zip(x, y))

    counter = Counter()
    vocab_tags_list = [idx for b, idx in datasets['target_names'].items()]

    # Randomise and Split train/val set
    dev_sample_index = -1 * int(config.dev_sample_fraction * float(len(y)))
    correct_splitting = True
    while correct_splitting:
        random.shuffle(data)
        if config.enable_char:
            x_shuffled, y_shuffled, char_ids_shuffled, word_lengths_shuffled = zip(*data)
            char_ids_shuffled = np.asanyarray(char_ids_shuffled)
            word_lengths_shuffled = np.asanyarray(word_lengths_shuffled)
        else:
            x_shuffled, y_shuffled = zip(*data)

        x_shuffled = np.asanyarray(x_shuffled)
        y_shuffled = np.asanyarray(y_shuffled)

        # Split train/test set
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        if config.enable_char:
            char_ids_train, char_ids_dev = char_ids_shuffled[:dev_sample_index], char_ids_shuffled[dev_sample_index:]
            word_lengths_train, word_lengths_dev = word_lengths_shuffled[:dev_sample_index], word_lengths_shuffled[
                                                                                             dev_sample_index:]
        # Comparing the tags using idx of tags in vocab_tags in the training data
        y_train_temp = np.argmax(y_train, 1)
        tags_counter = counter.count(y_train_temp, vocab_tags_list)
        correct_splitting = False

        # To cross check that no data will be missed out for training set when only have one data in the class
        for x in tags_counter:
            if x == 0.0:
                correct_splitting = True

    # x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    # y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    # Printing data details
    print("=======================================================")
    print("Data Details:")
    print('Train/Dev split = %d/%d' % (len(y_train), len(y_dev)))
    print('Train shape = ', x_train.shape)
    print('Dev shape = ', x_dev.shape)
    if config.enable_char:
        print('Char_ids_shape = ', char_ids_train.shape)
        print('word_lengths_shape = ', word_lengths_train.shape)
    print('Vocab_size = ', len(vocab_processor.vocabulary_))
    print('Sentence max words = ', config.doc_length)
    print("=======================================================")

    # Loading class_weights for training
    weights_array = data_helpers.calculate_weight(np.argmax(y, 1), datasets['target_names'])

    if config.transfer_learning:
        # load checkpoint of the pretrained model
        print("Loading weights from pre-trained model {}".format(config.pretrained_model_path))
        with tf.Session() as pre_sess:
            pre_saver = tf.train.import_meta_graph("{}.meta".format('transfer_learning_models/1412/model.weights/'))
            pre_saver.restore(pre_sess, 'transfer_learning_models/1412/model.weights/')
            var_names = [v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
            fw_kernel = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          'chars/bidirectional_rnn/fw/lstm_cell/kernel:0')
            fw_bias = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        'chars/bidirectional_rnn/fw/lstm_cell/bias:0')
            bw_kernel = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          'chars/bidirectional_rnn/bw/lstm_cell/kernel:0')
            bw_kernel = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          'chars/bidirectional_rnn/bw/lstm_cell/bias:0')
        print('Found variables:')
        for v in var_names:
            print(v)

    # Loading model
    checkpoint_file = tf.train.latest_checkpoint(config.checkpoint_dir)
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=config.allow_soft_placement,
          log_device_placement=config.log_device_placement,
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            if config.enable_char:
                cnn = model(config=config,
                            sequence_length=x_train.shape[1],
                            num_classes=y_train.shape[1],
                            vocab_size=len(vocab_processor.vocabulary_),
                            embedding_size=config.embedding_dim,
                            filter_sizes=config.filter_sizes,
                            num_filters_layer1=config.num_filters[0],
                            l2_reg_lambda=config.l2_reg_lambda,
                            weights_array=weights_array, nchars=len(datasets['vocab_chars'])
                            )
            else:
                cnn = model(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=config.embedding_dim,
                    filter_sizes=config.filter_sizes,
                    num_filters_layer1=config.num_filters[0],
                    l2_reg_lambda=config.l2_reg_lambda,
                    weights_array=weights_array)

            # restoring from the checkpoint file
            if config.checkpoint_dir != "":
                ckpt = tf.train.get_checkpoint_state(config.checkpoint_dir)
                tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            # Learning rate is 1E-3
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

            print("=======================================================")
            print("Writing to {}\n".format(config.out_dir))
            print("=======================================================")

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
            weighted_acc_summary = tf.summary.scalar("weighted_accuracy", cnn.weighted_accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, weighted_acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(config.out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary, weighted_acc_summary])
            dev_summary_dir = os.path.join(config.out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(config.out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=config.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(config.out_dir, "vocab"))

            # Printing word embedding information if there is any.
            if config.enable_word_embeddings and config.embedding_name is not None:
                print("{} = {}".format("WORD_EMBEDDINGS", config.embedding_name))
            else:
                print("{} = {}".format("WORD_EMBEDDINGS", "NONE"))

            print("{} = {}en = [10, 20, 30,".format("EMBEDDING DIMENSION", str(config.embedding_dim)))
            print("=======================================================")

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Loading Word Embeddings
            if config.enable_word_embeddings and config.embedding_name is not None:
                vocabulary = vocab_processor.vocabulary_
                init_w = None
                if config.embedding_name == 'word2vec':
                    # load embedding vectors from the word2vec
                    print("Load word2vec file {}".format(config.word2vec_path))
                    init_w = data_helpers.load_embedding_vectors_word2vec(vocabulary, config.word2vec_path, config.word2vec)
                    print("word2vec file has been loaded...")
                    print("=======================================================")

                elif config.embedding_name == 'glove':
                    # load embedding vectors from the glove
                    print("Load glove file {}".format(config.glove_path))
                    init_w = data_helpers.load_embedding_vectors_glove(vocabulary, config.glove_path, config.glove_dim)
                    print("glove file has been loaded\n")
                    print("=======================================================")
                sess.run(cnn.W.assign(init_w))

            if config.transfer_learning:
                print('Initializing bi-LSTM with pretrained-weights...')
                sess.run()

            #  Import checkpoint graph if want to continue training from checkpoint
            if config.checkpoint_dir != "":
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

            # training method
            def train_step(x_batch, y_batch, char_ids=None, word_lengths=None):
                """
                A single training step
                """
                if config.enable_char:
                    feed_dict = {cnn.input_x: x_batch,
                                 cnn.input_y: y_batch,
                                 cnn.dropout_keep_prob: config.dropout_keep_prob,
                                 cnn.word_lengths: word_lengths,
                                 cnn.char_ids: char_ids}
                else:
                    feed_dict = {cnn.input_x: x_batch,
                                 cnn.input_y: y_batch,
                                 cnn.dropout_keep_prob: config.dropout_keep_prob}
                _, step, summaries, loss, accuracy, weighted_accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.weighted_accuracy],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}, wacc {:g}\n".format(time_str, step, loss, accuracy, weighted_accuracy))

                train_summary_writer.add_summary(summaries, step)

            # Validation method
            def dev_step(x_batch, y_batch, word_lengths=None, char_ids=None, writer=None):
                """
                Evaluates model on a dev set
                """
                if config.enable_char:
                    feed_dict = {cnn.input_x: x_batch,
                                 cnn.input_y: y_batch,
                                 cnn.dropout_keep_prob: config.dropout_keep_prob,
                                 cnn.word_lengths: word_lengths,
                                 cnn.char_ids: char_ids}
                else:
                    feed_dict = {cnn.input_x: x_batch,
                                 cnn.input_y: y_batch,
                                 cnn.dropout_keep_prob: config.dropout_keep_prob}

                # JIALER MOD COMMENTED
                # step, summaries, loss, accuracy, weighted_accuracy = sess.run(
                #     [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.weighted_accuracy],
                #     feed_dict)

                step, summaries, loss, accuracy, weighted_accuracy, predictions = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.weighted_accuracy, cnn.predictions],
                    feed_dict)

                # Calculate p, r, f1 for each class
                y_shape = y_batch.shape[1]
                confusion_matrix = np.zeros([y_shape, y_shape])
                for a, b in zip(np.argmax(y_batch, 1), predictions):
                    confusion_matrix[a][b] += 1
                tp_by_tags = np.zeros(y_shape)
                total_predict_by_tags = np.zeros(y_shape)
                total_labeled_by_tags = np.zeros(y_shape)
                precision_by_tags = np.zeros(y_shape)
                recall_by_tags = np.zeros(y_shape)
                f1_by_tags = np.zeros(y_shape)

                for idx in range(y_shape):
                    tp_by_tags[idx] = confusion_matrix[idx][idx]
                    for n in range(y_shape):
                        total_labeled_by_tags[idx] += confusion_matrix[idx][n]
                        total_predict_by_tags[idx] += confusion_matrix[n][idx]
                    precision_by_tags[idx] = tp_by_tags[idx]/total_predict_by_tags[idx] if tp_by_tags[idx] > 0 else 0
                    recall_by_tags[idx] = tp_by_tags[idx]/total_labeled_by_tags[idx] if tp_by_tags[idx] > 0 else 0
                    f1_by_tags[idx] = (2 * precision_by_tags[idx] * recall_by_tags[idx])/(precision_by_tags[idx]+recall_by_tags[idx]) if tp_by_tags[idx] > 0 else 0

                average_f1 = np.average(f1_by_tags)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}, wacc {:g}, fscore {:g}\n".format(time_str, step, loss, accuracy, weighted_accuracy, average_f1))

                if writer:
                    writer.add_summary(summaries, step)
                return loss, average_f1

# ====================================================================================================================================================
            # Main code for training

            # Generating batches
            if config.enable_char:
                batches = data_helpers.batch_iter_lstm(x_train, y_train, char_ids_train, word_lengths_train,
                                                  config.batch_size, config.num_epochs)
            else:
                batches = data_helpers.batch_iter(list(zip(x_train, y_train)), config.batch_size, config.num_epochs)

            num_batches_per_epoch = int((len(x_train) - 1) / config.batch_size) + 1

            print("Training Starting.......")

            i = 0
            min_loss = 10000
            max_f_score = 0
            patience_cnt = 0
            hist_loss = []
            hist_f_score = []

            # Training loop. For each batch...
            if config.enable_char:
                for (x_batch, y_batch, char_ids_batch, word_lengths_batch) in batches:
                    train_step(x_batch, y_batch, char_ids_batch, word_lengths_batch)
                    current_step = tf.train.global_step(sess, global_step)

                    if current_step % num_batches_per_epoch == 0:
                        print("=======================================================")
                        print("Epoch number: {}".format(i + 1))
                        print("\nEvaluation:")
                        loss, f_score = dev_step(x_dev, y_dev, word_lengths_dev, char_ids_dev, writer=dev_summary_writer)
                        hist_loss.append(loss)
                        hist_f_score.append(f_score)
                        print("=======================================================")

                        # if i > 0 and hist_loss[i - 1] - hist_loss[i] > config.min_delta:
                        #     if hist_loss[i] < min_loss:
                        if i > 0 and hist_f_score[i] > max_f_score:
                            # min_loss = hist_loss[i]
                            max_f_score = hist_f_score[i]
                            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                            print("----new BEST f1 score----")
                            print("Saved model checkpoint to {}\n".format(path))
                            print("Checkpoint Number = {}".format(i + 1))
                            print("=======================================================")
                            patience_cnt = 0
                        else:
                            patience_cnt += 1
                        if patience_cnt > config.patience:
                            # if hist_loss[i] < min_loss:
                            print("Early stopping without at Epoch {} improvement.....".format(i + 1))
                            break
                        i += 1

            else:
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    train_step(x_batch, y_batch)

                    current_step = tf.train.global_step(sess, global_step)

                    if current_step % num_batches_per_epoch == 0:
                        print("=======================================================")
                        print("Epoch number: {}".format(i + 1))
                        print("\nEvaluation:")
                        loss, f_score = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                        hist_loss.append(loss)
                        hist_f_score.append(f_score)
                        print("=======================================================")

                        # if i > 0 and hist_loss[i - 1] - hist_loss[i] > config.min_delta:
                        #     if hist_loss[i] < min_loss:
                        if i > 0 and hist_f_score[i] > max_f_score:
                                # min_loss = hist_loss[i]
                            max_f_score = hist_f_score[i]
                            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                            print("----new BEST f1 score----")
                            print("Saved model checkpoint to {}\n".format(path))
                            print("Checkpoint Number = {}".format(i+1))
                            print("=======================================================")
                            patience_cnt = 0
                        else:
                            patience_cnt += 1
                        if patience_cnt > config.patience:
                            # if hist_loss[i] < min_loss:
                            print("Early stopping without at Epoch {} improvement.....".format(i+1))
                            break
                        i += 1
