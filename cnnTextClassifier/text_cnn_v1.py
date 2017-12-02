import tensorflow as tf
import numpy as np


class TextCNNv1(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters_layer1, num_filters_layer2=64, num_filters_layer3=None,
      l2_reg_lambda=0.0, weights_array=1.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        h_combined = []
        h_combined1 = []
        h_combined2 = []
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv0-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, num_filters_layer1]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters_layer1]), name="b")
                conv = tf.nn.conv1d(
                    self.embedded_chars,
                    W,
                    stride=1,
                    padding="SAME",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                h_combined.append(h)
        concat_h = tf.concat(h_combined, 2)

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv1-%s" % filter_size):
                # Second Convolution Layer
                # reshape_h = tf.squeeze(concat_h, axis=1)
                filter_shape = [filter_size, num_filters_layer1*3, num_filters_layer2]
                W1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W1")
                b1 = tf.Variable(tf.constant(0.1, shape=[num_filters_layer2]), name="b1")
                conv1 = tf.nn.conv1d(
                    concat_h, W1,
                    stride=1,
                    padding="SAME",
                    name="conv1"
                )
                # Apply nonlinearity
                h1 = tf.nn.relu(tf.nn.bias_add(conv1, b1), name="relu1")
                h_combined1.append(h1)

        # concat_h1 = tf.concat(h_combined1, 2)
        #
        # for i, filter_size in enumerate(filter_sizes):
        #     with tf.name_scope("conv2-%s" % filter_size):
        #         # Second Convolution Layer
        #         # reshape_h = tf.squeeze(concat_h, axis=1)
        #         filter_shape = [filter_size, num_filters_layer2*3, num_filters_layer3]
        #         W2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W1")
        #         b2 = tf.Variable(tf.constant(0.1, shape=[num_filters_layer3]), name="b1")
        #         conv2 = tf.nn.conv1d(
        #             concat_h1, W2,
        #             stride=1,
        #             padding="SAME",
        #             name="conv2"
        #         )
                # Apply nonlinearity
                # h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu2")
                # h_combined2.append(h2)

        final_h = tf.expand_dims(tf.concat(h_combined1, 2), -1)

        # Maxpooling over the outputs
        self.h_pool = tf.nn.max_pool(
            final_h,
            ksize=[1, 4, 1, 1],
            strides=[1, 2, 1, 1],
            padding='VALID',
            name="pool")

        # Combine all the pooled features
        flatten_size = num_filters_layer2 * len(filter_sizes) * int((((sequence_length-4)/2)+1))
        # self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, flatten_size])

        # Add Fully connected Layer
        with tf.name_scope("FC"):
            self.dense = tf.layers.dense(self.h_pool_flat, units=(flatten_size/2), activation=tf.nn.relu)
            # self.dense1 = tf.layers.dense(self.dense, units=(flatten_size/4), activation=tf.nn.relu)


        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.dense, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[flatten_size/2, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            # print("scores shape is " + str(self.scores.get_shape()))
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            class_weight = tf.expand_dims(tf.constant(weights_array), 1)
            weight = tf.reduce_sum(tf.matmul(self.input_y, class_weight), axis=1)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            weighted_loss = tf.multiply(losses, weight)
            self.loss = tf.reduce_mean(weighted_loss) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        with tf.name_scope("weighted_accuracy"):
            class_weight = tf.expand_dims(tf.constant(weights_array), 1)
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            transformed_correct_predictions = tf.expand_dims(tf.cast(correct_predictions, "float"), 1)
            # predictions_onehot = tf.one_hot(tf.argmax(self.input_y, 1), num_classes)
            weighted_correct_label_temp = tf.matmul(self.input_y, class_weight)
            # self.input_y_transformed = tf.argmax(self.input_y, 1)
            # weighted_correct_predictions_temp = tf.matmul(self.input_y, tf.constant(weights_array))
            # weighted_correct_predictions = tf.multiply(weighted_correct_predictions_temp, tf.cast(correct_predictions, "float"))
            weighted_correct_labels = tf.multiply(weighted_correct_label_temp, transformed_correct_predictions)
            # self.weighted_accuracy = tf.reduce_mean(weighted_correct_predictions, name="weighted_accuracy")
            self.weighted_accuracy = tf.divide(tf.reduce_sum(weighted_correct_labels),
                                               tf.reduce_sum(weighted_correct_label_temp), name="weighted_accuracy")

        # with tf.name_scope("precision"):
        #     # prediction_one_hot = tf.one_hot(self.predictions, num_classes)
        #     # total_prediction_tags = tf.reduce_sum(prediction_one_hot, 0)
        #     # correct_pred_tags_temp = tf.matmul(tf.expand_dims(tf.cast(correct_predictions, "float"), 0), prediction_one_hot)
        #     # correct_pred_tags = tf.reduce_sum(correct_pred_tags_temp, 0)
        #     self.precision = tf.metrics.precision(self.predictions, tf.argmax(self.input_y, 1))
        #     # correct_predictions_temp = tf.matmul(prediction_one_hot, class_weight)
        #     # self.weighted_precision = tf.divide(tf.reduce_sum(weighted_correct_labels),
        #     #                                     tf.reduce_sum(correct_predictions_temp),
        #     #                                     name="precision")
        #
        # with tf.name_scope("recall"):
        #     total_label_tags = tf.reduce_sum(self.input_y, 0)
        #     self.recall = tf.reduce_mean(tf.divide(correct_pred_tags, total_label_tags), name='recall')
        #
        # with tf.name_scope("f1"):
        #     self.fscore = tf.divide(tf.multiply(2.0, tf.multiply(self.recall, self.precision)),
        #                                  tf.add(self.precision, self.recall), name="fscore")
