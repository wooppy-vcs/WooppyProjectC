import tensorflow as tf
import numpy as np
import os

import yaml
from tensorflow.contrib import learn
import argparse
from cnnTextClassifier import data_helpers


from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template, jsonify, Response, \
    json
from flask_restful import Resource, Api

from cnnTextClassifier.config import Config
from cnnTextClassifier.data_helpers import load_vocab


class OngoingSession:
    def __init__(self, checkpoint_dir, config):

        self.categories = load_vocab(config.tags_vocab_path)

        # Map data into vocabulary
        self.vocab_path = os.path.join(checkpoint_dir, "..", "vocab")
        self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(self.vocab_path)

        self.checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session_conf = tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False)
            self.sess = tf.Session(config=self.session_conf)

            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(self.checkpoint_file))
            saver.restore(self.sess, self.checkpoint_file)

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        if x.ndim == 1:
            x = x.reshape((1, -1))
        max_x = np.max(x, axis=1).reshape((-1, 1))
        exp_x = np.exp(x - max_x)
        return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

    def predict(self, x_raw):

        x_raw = [data_helpers.clean_str(x_raw)]
        x_test = np.array(list(self.vocab_processor.transform(x_raw)))

        # Get the placeholders from the graph by name
        input_x = self.graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = self.graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        scores = self.graph.get_operation_by_name("output/scores").outputs[0]

        # Tensors we want to evaluate
        predictions = self.graph.get_operation_by_name("output/predictions").outputs[0]

        # Collect the predictions here

        predictions, scores = self.sess.run([predictions, scores],
                                            {input_x: x_test, dropout_keep_prob: 1.0})
        prediction = predictions[0]
        probabilities = self.softmax(scores)[0]

        return prediction, probabilities

    def predict_words(self, words):
        prediction, probabilities = self.predict(words)

        idx_new_list = sorted(range(len(probabilities)), key=lambda k: probabilities[k])

        inv_categories = {v: k for k, v in self.categories.items()}
        predicted_category = None
        # tags = [str(inv_categories[x].ljust(25)) for x in idx_new_list]
        # prob = [str(probabilities[x]) for x in idx_new_list]
        tags_probabilities_pair = {}
        for idx_new in idx_new_list:
            # print(inv_categories[idx_new] + str("%.4f" % probabilities[idx_new]))
            tags_probabilities_pair[inv_categories[idx_new]] = str("%.4f" % probabilities[idx_new])
            if idx_new == prediction:
                predicted_category = inv_categories[idx_new]

        return json.dumps({"words": words, "category": predicted_category,
                           "tags": tags_probabilities_pair})

app = Flask(__name__, static_folder=os.path.join("templates", "assets"))
app.secret_key = 'rem4lyfe'
checkpoint_dir = "Enriched-runs/Scenario-len80-CNN/checkpoints"
config = Config()
ongoing_session = OngoingSession(checkpoint_dir, config)


@app.route('/get_tags', methods=['POST'])
def get_tags():
    if 'text' not in request.form:
        return '{"error" : "Add text to your request!"}'
    else:
        text = request.form['text']

    if text.strip() == '':
        return '{"error" : "Add non-whitespace text to your request!"}'

    my_dict_string = ongoing_session.predict_words(text.strip())
    return my_dict_string, 200, {'Content-Type': 'application/json; charset=utf-8'}

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port="1488")
