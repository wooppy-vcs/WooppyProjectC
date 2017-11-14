import random
import yaml
from cnnTextClassifier import data_helpers
#
# tf.flags.DEFINE_float("test_sample_percentage", .2, "Percentage of the training data to use for test")
# tf.flags.DEFINE_integer("sentences_column", 3, "Column number of sentence data in data txt file")
# tf.flags.DEFINE_integer("tags_column", 11, "Column number of tags in data txt file")

test_sample_percentage = 0.2
sentences_column = 3
tags_column = 11
# FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

datasets = data_helpers.get_datasets_multiple_files(container_path=cfg["datasets"]["datalocalfile"]["data_folder"]["path"],
                                                    sentences=sentences_column, tags=tags_column)

x_text, y = [datasets["data"], datasets["target"]]
print("Total data size : {}".format(len(x_text)))

# Randomly shuffle data
data = list(zip(x_text, y))
random.shuffle(data)
x_shuffled, y_shuffled = zip(*data)

# Split train/test set
test_sample_index = -1 * int(test_sample_percentage * float(len(y)))
x_train, x_test = x_shuffled[:test_sample_index], x_shuffled[test_sample_index:]
y_train, y_test = y_shuffled[:test_sample_index], y_shuffled[test_sample_index:]


data_helpers.write_data_to_file(x_train, y_train, "data/Project-A-R/Training_data.txt")
data_helpers.write_data_to_file(x_test, y_test, "data/Project-A-R/Test_data.txt")


