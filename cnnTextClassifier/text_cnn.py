import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, weights_array=1.0):
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
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
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
            # weighted_correct_predictions_temp = tf.matmul(self.input_y, tf.constant(weights_array))
            # weighted_correct_predictions = tf.multiply(weighted_correct_predictions_temp, tf.cast(correct_predictions, "float"))
            weighted_correct_labels = tf.multiply(weighted_correct_label_temp, transformed_correct_predictions)
            # self.weighted_accuracy = tf.reduce_mean(weighted_correct_predictions, name="weighted_accuracy")
            self.weighted_accuracy = tf.divide(tf.reduce_sum(weighted_correct_labels), tf.reduce_sum(weighted_correct_label_temp), name="weighted_accuracy")
            prediction_one_hot = tf.one_hot(self.predictions, num_classes)
            weighted_correct_predictions_temp = tf.matmul(prediction_one_hot, class_weight)
            self.weighted_precision = tf.divide(tf.reduce_sum(weighted_correct_labels), tf.reduce_sum(weighted_correct_predictions_temp), name="weighted_precision")
            self.weighted_f1 = tf.divide(tf.multiply(2, tf.multiply(self.weighted_accuracy, self.weighted_precision)), tf.add(self.weighted_precision, self.weighted_accuracy), name="weighted_f1")
