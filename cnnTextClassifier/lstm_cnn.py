import tensorflow as tf

class LSTMCNN(object):

    def __init__(self, config, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters_layer1, num_filters_layer2=None, num_filters_layer3=None,
      l2_reg_lambda=0.0, weights_array=1.0, nchars=None):
        self.config = config
        self.nchars = nchars
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                                           name="word_lengths")
        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                                       name="char_ids")
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        with tf.variable_scope("chars"):
            # get embeddings matrix
            _char_embeddings = tf.get_variable(name="_char_embeddings", dtype=tf.float32,
                shape=[self.nchars, self.config.dim_char])
            char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.char_ids,
                name="char_embeddings")
            # put the time dimension on axis=1
            s = tf.shape(char_embeddings)
            char_embeddings = tf.reshape(char_embeddings, shape=[-1, s[-2], self.config.dim_char])
            char_embeddings = tf.Print(char_embeddings, [char_embeddings])
            word_lengths = tf.reshape(self.word_lengths, shape=[-1])
            # bi lstm on chars
            # need 2 instances of cells since tf 1.1
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.char_hidden_size, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.char_hidden_size, state_is_tuple=True)

            _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, char_embeddings,
                                                                                  sequence_length=word_lengths,
                                                                                  dtype=tf.float32)

            output = tf.concat([output_fw, output_bw], axis=-1)
            # shape = (batch size, max sentence length, char hidden size)
            output = tf.reshape(output, shape=[-1, s[1], 2*self.config.char_hidden_size])

            word_embeddings = tf.concat([self.embedded_chars, output], axis=-1)

        # Add dropout
        word_embeddings = tf.nn.dropout(word_embeddings, self.config.dropout_keep_prob)
        self.word_embeddings = tf.expand_dims(word_embeddings, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size+(config.dim_char*2), 1, num_filters_layer1]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters_layer1]), name="b")
                conv = tf.nn.conv2d(
                    self.word_embeddings,
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
        num_filters_total = num_filters_layer1 * len(filter_sizes)
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
            self.weighted_accuracy = tf.divide(tf.reduce_sum(weighted_correct_labels),
                                               tf.reduce_sum(weighted_correct_label_temp), name="weighted_accuracy")
