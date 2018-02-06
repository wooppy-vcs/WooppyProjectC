import csv

import numpy as np

from cnnTextClassifier.REST_server.main_config import REST_Config
from cnnTextClassifier.REST_server.ternary_class_rest_main import OngoingSession
from cnnTextClassifier.data_helpers import load_vocab
from cnnTextClassifier.regex_tagger import create_persistent_dictionaries, tag_input_string
from cnnTextClassifier.REST_server.ternary_class_rest_main import prepare_vocab


class RegexTernaryModel:
    def __init__(self, keyword_dict_compiled_pattern, scenario_dictionary, ongoing_session):
        self.keyword_dict_compiled_pattern = keyword_dict_compiled_pattern
        self.scenario_dictionary = scenario_dictionary
        self.ongoing_session = ongoing_session

    def predict_prioritise_model(self, sentence):
        regex_tag = tag_input_string(sentence, keyword_dict_compiled_pattern, scenario_dictionary)
        model_tag, level1_probabilities, tags_probabilities_pair = self.ongoing_session.predict_sentence(sentence)
        if model_tag == "None" and float(level1_probabilities["None"]) > 0.8:
            return regex_tag, "regex", level1_probabilities, tags_probabilities_pair
        else:
            return model_tag, "model", level1_probabilities, tags_probabilities_pair

    def predict_prioritise_regex(self, sentence):
        regex_tag = tag_input_string(sentence, keyword_dict_compiled_pattern, scenario_dictionary)
        model_tag = self.ongoing_session.predict_sentence(sentence)
        if regex_tag == "None":
            return model_tag, "model"
        else:
            return regex_tag, "regex"

    def run_evaluation(self, test, vocab_tags, test_vocab_map):
        y_shape = len(vocab_tags)
        confusion_matrix = np.zeros([y_shape, y_shape])
        tp_by_tags = np.zeros(y_shape)
        total_predict_by_tags = np.zeros(y_shape)
        total_labeled_by_tags = np.zeros(y_shape)
        precision_by_tags = np.zeros(y_shape)
        recall_by_tags = np.zeros(y_shape)
        f1_by_tags = np.zeros(y_shape)
        predictions = []
        tag_sources = []
        # probabilities_list = []
        # idx_to_tag = {idx: tag for tag, idx in vocab_tags.items()}
        probabilities = []
        i = 1
        for sentence, tag in test:
            print("Data Number: {}".format(i))
            prediction, tag_source, l1_probs, scenario_probs = self.predict_prioritise_model(sentence)
            confusion_matrix[vocab_tags[test_vocab_map[tag]]][vocab_tags[test_vocab_map[prediction]]] += 1
            predictions.append(test_vocab_map[prediction])
            tag_sources.append(tag_source)
            probabilities.append(scenario_probs)
            i += 1
        print("Evaluating...")
        for idx in range(y_shape):
            tp_by_tags[idx] = confusion_matrix[idx][idx]
            for n in range(y_shape):
                total_labeled_by_tags[idx] += confusion_matrix[idx][n]
                total_predict_by_tags[idx] += confusion_matrix[n][idx]
            recall_by_tags[idx] = tp_by_tags[idx] / total_labeled_by_tags[idx] if tp_by_tags[idx] > 0 else 0
            precision_by_tags[idx] = tp_by_tags[idx] / total_predict_by_tags[idx] if tp_by_tags[idx] > 0 else 0
            f1_by_tags[idx] = (2 * precision_by_tags[idx] * recall_by_tags[idx]) / (
                precision_by_tags[idx] + recall_by_tags[idx]) if tp_by_tags[idx] > 0 else 0

        return precision_by_tags, recall_by_tags, f1_by_tags, predictions, confusion_matrix, tag_sources, probabilities

    def evaluate(self, test_data, vocab_tags, test_vocab_map):
        precision_by_tags, recall_by_tags, f1_by_tags, predictions, confusion_matrix, tag_sources, scenario_probs = \
            self.run_evaluation(test_data, vocab_tags, test_vocab_map)
        sentences, tags_label = zip(*test_data)
        tags_label = [test_vocab_map[tag] for tag in tags_label]

        human_readable_results = np.column_stack((np.array(sentences), predictions, tags_label, tag_sources))
        print("Generating Reports...")
        generate_results("Runs/regex_model_results_0.1.2.csv",
                         human_readable_results)
        generate_report("Runs/regex_model_report_0.1.2.txt", precision_by_tags, recall_by_tags,
                        f1_by_tags, vocab_tags)
        print("End of Test...")


def generate_results(path, human_readable):
    """
    Takes in results and write them into readable txt file.
    :param config: config for the trained models
    :param human_readable: numpy stacked column which stores data about the results
    :return:
    """
    headers = ["Input", "Predicted", "Expected", "From"]
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow(headers)
        csv.writer(f).writerows(human_readable)
    f.close()


def generate_report(path, p_bytags, r_bytags, f1_bytags, tags):
    """
    Takes in the metric report of the evaluation and write it into readable txt file
    :param config: config for the trained models
    :param tp: true positive by class calculated, list of int32
    :param total_correct_tags: total labeled tags by class, list of int32
    :param total_pred_tags: total models predicted tags by class, list of int32
    :param tags: dic[idx] = tag
    :return:
    """
    with open(path, 'w', newline='') as f:
        f.write("tags\tPrecision\tRecall\tF1\n")
        for name, idx in tags.items():
            f.write("{}\t{:04.2f}\t{:04.2f}\t{:04.2f}\n".format(name, p_bytags[idx], r_bytags[idx],
                                                                f1_bytags[idx]))
    f.close()


def read_test_set(path):
    data = list(open(path, 'r', encoding="utf8").readlines())
    data = [s.split("\t") for s in data]
    sentences = [s[0] for s in data]
    tags = [s[1].strip() for s in data]
    test = list(zip(sentences, tags))
    return test


def load_dict(path):
    example = list(open(path, 'r', encoding="utf8").readlines())
    f = [s.split("\t") for s in example]
    originals = [s[0] for s in f]
    maps = [s[1].strip() for s in f]
    merged_vocab = {original: mapped for original, mapped in zip(originals, maps)}
    return merged_vocab

if __name__ == "__main__":
    keyword_dict_compiled_pattern, scenario_dictionary = create_persistent_dictionaries()
    config = REST_Config()
    ongoing_session = OngoingSession(config, prepare_vocab)
    model = RegexTernaryModel(keyword_dict_compiled_pattern, scenario_dictionary, ongoing_session)
    vocab_tags = load_vocab("data/Architecture-v2/v3-corrected-tags/new_vocab_tags.txt")
    test_vocab_map = load_dict("data/Architecture-v2/v3-corrected-tags/merged_tags.txt")
    test = read_test_set("data/Project-A-R-Scenario_Billing_Account-v2/Test_data.txt")

    model.evaluate(test_data=test, vocab_tags=vocab_tags, test_vocab_map=test_vocab_map)

    #
    # print(model.predict_prioritise_model("Why bar me?"))
