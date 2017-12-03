import random
import yaml
from cnnTextClassifier import data_helpers
#
# tf.flags.DEFINE_float("test_sample_percentage", .2, "Percentage of the training data to use for test")
# tf.flags.DEFINE_integer("sentences_column", 3, "Column number of sentence data in data txt file")
# tf.flags.DEFINE_integer("tags_column", 11, "Column number of tags in data txt file")
from cnnTextClassifier.config import Config
from cnnTextClassifier.data_analyser import Counter
from cnnTextClassifier.data_helpers import calculate_weight

test_sample_percentage = 0.2
# val_sample_percentage = 0.1 # 20% of training data
sentences_column = 3

# for Level-1
# tags_column = 11

# for Level-2
# tags_column = 12

# for AnswerType
# tags_column = 13

# for Confirmation
# tags_column = 14

# for Scenario
tags_column = 15


config = Config()
datasets = data_helpers.get_datasets_multiple_files(container_path=cfg["datasets"]["datalocalfile"]["data_folder"]["path"],
                                                    vocab_tags_path=cfg["datasets"]["localfile"]["vocab_write_path"]["path"],
                                                    # class_weights_path=cfg["datasets"]["localfile"]["class_weights_path"]["path"],
                                                    system_path=cfg["datasets"]["localfile"]["container_path"],
                                                    sentences=sentences_column, tags=tags_column, remove_none=False)

x_raw, y_raw = datasets["data"], datasets["target"]

print("Total data size : {}".format(len(x_raw)))

vocab_char = data_helpers.get_char_vocab(x_raw)

# Loading tags dictionary and change tag names in y to numbers
vocab_tags = data_helpers.load_vocab(config.tags_vocab_path)
vocab_tags_list = [b for b, idx in vocab_tags.items()]
# target = []
# for s in y_raw:
#         target.append(int(vocab_tags[str(s)]))
counter = Counter()

correct_splitting = True

data = list(zip(x_raw, y_raw))
x_train, x_test = [], []
y_train, y_test = [], []
test_sample_index = -1 * (int(test_sample_percentage * float(len(y_raw))))

while correct_splitting:
    # Randomly shuffle data
    random.shuffle(data)
    x_shuffled, y_shuffled = zip(*data)
    # Split train/test set
    x_train, x_test = x_shuffled[:test_sample_index], x_shuffled[test_sample_index:]
    y_train, y_test = y_shuffled[:test_sample_index], y_shuffled[test_sample_index:]

    tags_counter = counter.count(y_train, vocab_tags_list)
    correct_splitting = False
    for x in tags_counter:
        if x == 0.0:
            correct_splitting = True


data_helpers.write_data_to_file(x_train, y_train, config.training_path)
data_helpers.write_data_to_file(x_test, y_test, config.test_path)
data_helpers.write_vocab_tags(vocab_char, config.char_vocab_path)


