import tensorflow as tf


class ImportGraph():
    """  Importing and running isolated TF graph """
    def __init__(self, loc):
        # Create local graph and use it in the session
        self.checkpoint_file = tf.train.latest_checkpoint(loc)
        self.graph = tf.Graph()
        self.session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(graph=self.graph, config=self.session_conf)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            saver = tf.train.import_meta_graph("{}.meta".format(self.checkpoint_file),
                                               clear_devices=True)
            saver.restore(self.sess, self.checkpoint_file)
            # Get activation function from saved collection
            # You may need to change this in case you name it differently
            # Tensors we want to evaluate
            self.scores = self.graph.get_operation_by_name("output/scores").outputs[0]

            # Tensors we want to evaluate
            self.predictions = self.graph.get_operation_by_name("output/predictions").outputs[0]

    def run(self, x_raw):
        """ Running the activation function previously imported """
        # The 'x' corresponds to name of input placeholder

        # Get the placeholders from the graph by name
        input_x = self.graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = self.graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        return self.sess.run([self.predictions, self.scores], feed_dict={input_x: x_raw,
                                                                         dropout_keep_prob: 1.0})
