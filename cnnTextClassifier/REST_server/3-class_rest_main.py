import os

import numpy as np
from tensorflow.contrib.learn.python.learn import preprocessing
from cnnTextClassifier.REST_server.main_config import REST_Config
from flask import Flask, request, json

from cnnTextClassifier import data_helpers
from cnnTextClassifier.REST_server.graph_importer import ImportGraph
from cnnTextClassifier.data_helpers import load_vocab


class OngoingSession:
    def __init__(self, config, prepare_vocab):

        self.l1_categories, self.l1_vocab_processor = prepare_vocab(config.l1_tags_vocab, config.l1_checkpoint_dir)
        self.account_categories, self.account_vocab_processor = prepare_vocab(config.account_tags_vocab,
                                                                              config.account_checkpoint_dir)
        self.billing_categories, self.billing_vocab_processor = prepare_vocab(config.billing_tags_vocab,
                                                                              config.billing_checkpoint_dir)

        self.L1_model = ImportGraph(config.l1_checkpoint_dir)
        self.account_model = ImportGraph(config.account_checkpoint_dir)
        self.billing_model = ImportGraph(config.billing_checkpoint_dir)

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        if x.ndim == 1:
            x = x.reshape((1, -1))
        max_x = np.max(x, axis=1).reshape((-1, 1))
        exp_x = np.exp(x - max_x)
        return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

    def predict(self, x_raw, model, vocab_processor):

        x_raw = [data_helpers.clean_str(x_raw)]

        x_test = np.array(list(vocab_processor.transform(x_raw)))

        # Collect the predictions here
        predictions, scores = model.run(x_test)
        prediction = predictions[0]
        probabilities = self.softmax(scores)[0]

        return prediction, probabilities

    def evaluation(self, words, model, vocab_processor, categories):
        prediction, probabilities = self.predict(words, model, vocab_processor)

        idx_new_list = sorted(range(len(probabilities)), key=lambda k: probabilities[k])

        inv_categories = {v: k for k, v in categories.items()}
        predicted_category = None
        # tags = [str(inv_categories[x].ljust(25)) for x in idx_new_list]
        # prob = [str(probabilities[x]) for x in idx_new_list]
        tags_probabilities_pair = {}
        for idx_new in idx_new_list:
            # print(inv_categories[idx_new] + str("%.4f" % probabilities[idx_new]))
            tags_probabilities_pair[inv_categories[idx_new]] = str("%.4f" % probabilities[idx_new])
            if idx_new == prediction:
                predicted_category = inv_categories[idx_new]

        return predicted_category, tags_probabilities_pair

    def predict_words(self, words):
        predicted_category=None
        tags_probabilities_pair = {}

        level_1_prediction, level1_probabilities = self.evaluation(words, self.L1_model, self.l1_vocab_processor,
                                                                   self.l1_categories)

        if level_1_prediction == "Account":
            predicted_category, tags_probabilities_pair = self.evaluation(words, self.account_model,
                                                                          self.account_vocab_processor,
                                                                          self.account_categories)
        elif level_1_prediction == "Billing":
            predicted_category, tags_probabilities_pair = self.evaluation(words, self.billing_model,
                                                                          self.billing_vocab_processor,
                                                                          self.billing_categories)
        else:
            predicted_category = level_1_prediction
            tags_probabilities_pair = level1_probabilities

        return json.dumps({"words": words, "category": predicted_category,
                           "tags": tags_probabilities_pair})


def prepare_vocab(tags_vocab_path, checkpoint_dir):
    categories = load_vocab(tags_vocab_path)

    # Map data into vocabulary
    vocab_path = os.path.join(checkpoint_dir, "..", "vocab")
    vocab_processor = preprocessing.VocabularyProcessor.restore(vocab_path)

    return categories, vocab_processor


app = Flask(__name__, static_folder=os.path.join("templates", "assets"))
app.secret_key = 'rem4lyfe'
# checkpoint_dir = "Enriched-x10-runs(LSTM&CNNv0)/runs-0-Scenario-len80-CNN-Enriched/checkpoints"
config = REST_Config()
ongoing_session = OngoingSession(config, prepare_vocab)


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


@app.route('/get_model', methods=['GET'])
def get_model():
    string = json.dumps({"First Layer Model": ongoing_session.L1_model.checkpoint_file,
                         "Second Layer Account Model": ongoing_session.account_model.checkpoint_file,
                         "Second Layer Billing Model": ongoing_session.billing_model.checkpoint_file})
    return string, 200, {'Content-Type': 'application/json; charset=utf-8'}

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port="4444")
