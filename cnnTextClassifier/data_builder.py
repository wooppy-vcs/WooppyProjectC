import random

import numpy as np
import yaml
from cnnTextClassifier import data_helpers
#
# tf.flags.DEFINE_float("test_sample_percentage", .2, "Percentage of the training data to use for test")
# tf.flags.DEFINE_integer("sentences_column", 3, "Column number of sentence data in data txt file")
# tf.flags.DEFINE_integer("tags_column", 11, "Column number of tags in data txt file")
from cnnTextClassifier.config import Config
from cnnTextClassifier.data_analyser import Counter
from cnnTextClassifier.data_helpers import calculate_weight


def build_data(config, scenario, l_1, sentences_column, tags_column, test_sample_percentage):
    if scenario:
        # To map wrong tags to correct ones
        dict = {"DoublePayment": "HowRefund",
                "UseWhatAccountID": "UseWhatAccount",
                "PornInOthers": "PortInOthers",
                "WhenPaymentFirstDay": "WhenBillUpdate",
                "Others": "BillingOthers",
                "ConfirmPayment": "ConfirmPaymentReceived",
                "Procedure": "HowPay",
                "WhyBillAMount": "WhyBillAmount",
                "CheckunbarTime": "CheckUnbarTime",
                "HowChangeplan": "HowChangePlan",
                "HowCHangePlan": "HowChangePlan",
                "CheckUnbarTIme": "CheckUnbarTime",
                "MethodExcept": "HowPay",
                "HowChangeDetails": "ChangeDetails"}

    if l_1:
        dict = {"Wrong": "None"}

    datasets = data_helpers.get_datasets_multiple_files(container_path=config.default_raw_data_path,
                                                        vocab_tags_path=config.tags_vocab_path,
                                                        vocab_char_path=config.char_vocab_path, system_path=config.out_dir,
                                                        sentences=sentences_column, tags=tags_column, remove_none=False)

    x_raw, y_raw = datasets["data"], datasets["target"]
    vocab_char, vocab_tags = datasets['vocab_chars'], datasets['vocab_tags']



    i = 0
    for x in y_raw:
        for y in dict:
            if x == y:
                y_raw[i] = dict[y]
        i += 1

    for n in dict:
        vocab_tags.remove(n)

    print("Total data size : {}".format(len(x_raw)))

    # # Generating chars vocabulary
    # vocab_char = data_helpers.get_char_vocab(x_raw)

    # # Loading tags dictionary and change tag names in y to numbers
    # vocab_tags = data_helpers.load_vocab(config.tags_vocab_path)
    vocab_tags_list = list(vocab_tags)

    counter = Counter()

    correct_splitting = True

    data = list(zip(x_raw, y_raw))
    x_train, x_test = [], []
    y_train, y_test = [], []
    test_sample_index = -1 * (int(test_sample_percentage * float(len(y_raw))))

    j = 0

    while correct_splitting:
        # Randomly shuffle data
        random.shuffle(data)
        x_shuffled, y_shuffled = zip(*data)
        # Split train/test set
        x_train, x_test = x_shuffled[:test_sample_index], x_shuffled[test_sample_index:]
        y_train, y_test = y_shuffled[:test_sample_index], y_shuffled[test_sample_index:]

        tags_counter = counter.count(y_train, vocab_tags_list)
        correct_splitting = False

        print("split number: {}".format(j))

        j += 1
        # To cross check that no data will be missed out for training set when only have one data in the class
        for x in tags_counter:
            if x == 0.0:
                correct_splitting = True

    data_helpers.write_data_to_file(x_train, y_train, config.training_path)
    data_helpers.write_data_to_file(x_test, y_test, config.test_path)
    data_helpers.write_vocab_tags(vocab_char, config.char_vocab_path)
    data_helpers.write_vocab_tags(vocab_tags, config.tags_vocab_path)


def converting_data(data_path, test_path, binary, outpath, dev_percentage=0.2, remove_none=False, dict_path=""):
    if binary:
        examples = list(open(data_path, 'r', encoding="utf8").readlines())
        examples = [s.split("\t") for s in examples]
        train_sentences = [s[0] for s in examples]
        train_tags = [s[1].strip() for s in examples]
        if remove_none:
            new_train_sentences = []
            new_train_tags = []
            data = list(zip(train_sentences, train_tags))
            dev_sample_index = -1 * int(dev_percentage * float(len(train_tags)))
            random.shuffle(data)
            x_shuffled, y_shuffled = zip(*data)
            x_shuffled = np.asanyarray(x_shuffled)
            y_shuffled = np.asanyarray(y_shuffled)
            x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
            y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

            for idx, tag in enumerate(y_dev):
                if tag != "None":
                    y_dev[idx] = "WithScenarios"
                else:
                    y_dev[idx] = "WithoutScenarios"

        if remove_none:
            for idx, tag in enumerate(y_train):
                if tag != "None":
                    new_train_sentences.append(x_train[idx])
                    new_train_tags.append("WithScenarios")
        else:
            for idx, tag in enumerate(train_tags):
                if tag != "None":
                    train_tags[idx] = "WithScenarios"
                else:
                    train_tags[idx] = "WithoutScenarios"

        if remove_none:
            data_helpers.write_data_to_file(new_train_sentences, new_train_tags, outpath+"/Training_data.txt")
            data_helpers.write_data_to_file(x_dev, y_dev, outpath+"/Development_data.txt")
        else:
            data_helpers.write_data_to_file(train_sentences, train_tags, outpath + "/Training_data.txt")


        if remove_none:
            tags_vocab = data_helpers.get_vocab_tags(new_train_tags)
            tags_vocab.add("WithoutScenarios")
        else:
            tags_vocab = data_helpers.get_vocab_tags(train_tags)
        data_helpers.write_vocab_tags(tags_vocab, outpath + "/tags_vocab.txt")

        test = list(open(test_path, 'r', encoding="utf8").readlines())
        test = [s.split("\t") for s in test]
        test_sentences = [s[0] for s in test]
        test_tags = [s[1].strip() for s in test]

        for idx, tag in enumerate(test_tags):
            if tag != "None":
                test_tags[idx] = "WithScenarios"
            else:
                test_tags[idx] = "WithoutScenarios"

        data_helpers.write_data_to_file(test_sentences, test_tags, outpath + "/Test_data.txt")
    else:
        mapping = dict()
        maps = list(open(dict_path, 'r', encoding="utf8").readlines())
        maps = [s.split("\t") for s in maps]
        for x, y in maps:
            mapping[x.strip()] = y.strip()
        # Load data from files
        examples = list(open(data_path, 'r', encoding="utf8").readlines())
        examples = [s.split("\t") for s in examples]
        train_sentences = [s[0] for s in examples]
        train_tags = [s[1].strip() for s in examples]

        if remove_none:
            new_train_sentences = []
            new_train_tags = []
        # remove none in this case is to remove none but retain the L3 scenarios
        #  for remove_none=False case, it will be L3 map to L1

        for idx, tag in enumerate(train_tags):
            if remove_none:
                if tag != "None":
                    new_train_sentences.append(train_sentences[idx])
                    new_train_tags.append(tag)
            else:
                train_tags[idx] = mapping[tag]

        if remove_none:
            data_helpers.write_data_to_file(new_train_sentences, new_train_tags, outpath + "/Training_data.txt")
        else:
            data_helpers.write_data_to_file(train_sentences, train_tags, outpath + "/Training_data.txt")

        if remove_none:
            tags_vocab = data_helpers.get_vocab_tags(new_train_tags)
        else:
            tags_vocab = data_helpers.get_vocab_tags(train_tags)
        data_helpers.write_vocab_tags(tags_vocab, outpath+"/tags_vocab.txt")

        test = list(open(test_path, 'r', encoding="utf8").readlines())
        test = [s.split("\t") for s in test]
        test_sentences = [s[0] for s in test]
        test_tags = [s[1].strip() for s in test]

        if remove_none:
            new_test_sentences = []
            new_test_tags = []

        for idx, tag in enumerate(test_tags):
            if remove_none:
                if tag != "None":
                    new_test_sentences.append(test_sentences[idx])
                    new_test_tags.append(tag)
            else:
                test_tags[idx] = mapping[tag]

        if remove_none:
            data_helpers.write_data_to_file(new_test_sentences, new_test_tags, outpath + "/Test_data.txt")
        else:
            data_helpers.write_data_to_file(test_sentences, test_tags, outpath + "/Test_data.txt")


def split_bill_account(dict_path, data_path, outpath_bill, outpath_account):
    """
    Splitting dataset into billing and account data plus removing None followed by storing them into
    respective directories.
    :param dict_path: L3:L1 map dictionary
    :param data_path: data that needs to be splitted
    :param outpath_bill: directory to store bill only dataset
    :param outpath_account: directory to store account only dataset
    :return: no return.
    """
    mapping = dict()
    maps = list(open(dict_path, 'r', encoding="utf8").readlines())
    maps = [s.split("\t") for s in maps]
    for x, y in maps:
        mapping[x.strip()] = y.strip()
    # Load data from files
    examples = list(open(data_path, 'r', encoding="utf8").readlines())
    examples = [s.split("\t") for s in examples]
    train_sentences = [s[0] for s in examples]
    train_tags = [s[1].strip() for s in examples]

    bill_sentences = []
    bill_tags = []
    account_sentences = []
    account_tags = []

    for idx, tag in enumerate(train_tags):
        if tag != "None":
            if mapping[tag] == "Account":
                account_sentences.append(train_sentences[idx])
                account_tags.append(tag)
            elif mapping[tag] == "Billing":
                bill_sentences.append(train_sentences[idx])
                bill_tags.append(tag)

    data_helpers.write_data_to_file(account_sentences, account_tags, outpath_account)
    data_helpers.write_data_to_file(bill_sentences, bill_tags, outpath_bill)


def convert_l1_to_l3(test_set, l3_test_dict, outpath):

    mapping = dict()
    maps = list(open(l3_test_dict, 'r', encoding="utf8").readlines())
    maps = [s.split("\t") for s in maps]
    for x, y in maps:
        mapping[x.strip()] = y.strip()

    tags = []

    examples = list(open(test_set, 'r', encoding="utf8").readlines())
    sentences = [s.strip() for s in examples]

    for sentence in sentences:
        tags.append(mapping[sentence])

    data_helpers.write_data_to_file(sentences, tags, outpath)

# ================================Build Data=======================================================
# test_sample_percentage = 0.2
# # val_sample_percentage = 0.1 # 20% of training data
# sentences_column = 3
#
# # for Level-1
# # tags_column = 11
#
# # for Level-2
# # tags_column = 12
#
# # for AnswerType
# # tags_column = 13
#
# # for Confirmation
# # tags_column = 14
#
# # for Scenario
# tags_column = 15
#
#
# scenario = True
# l_1 = False
#
# config = Config(dataset_name="Project-A-R-Scenario_Billing_Account-v3")
# build_data(config, True, False, sentences_column, tags_column, test_sample_percentage)


# ==================================== converting data ===============================================
# dict_path = "data/L3-Map-None.txt"
# dict_path = "data/Architecture-v2/v4/L3-Map-ReducedtoOthers.txt"
# dict_path = "data/Architecture-v2/L3-Map-L1-Dict.txt"

# outpath = "Project-A-R-Scenario-Acc-Bill-Reduced-Class"
# outpath = "Project-A-R-Scenario-Acc-Bill-ReducedtoOthers"
# outpath = "Architecture-v2/v2/Project-A-R-Level-1-Reduced-Class"
# outpath = "Architecture-v2/v4/Project-A-R-Level-1-ReducedtoOthers"

datapath = "data/Project-A-R-Scenario-Acc-Bill-ReducedtoOthers"
# datapath = "data/Project-A-R-Scenario_Billing_Account-v2"
# datapath = "data/Project-A-R-Scenario-Acc-Bill-ReducedtoOthers"

# converting_data(datapath+"/Training_data.txt", datapath+"/Test_data.txt", binary=False, outpath="data/"+outpath,
#                 remove_none=False, dict_path=dict_path)


# ==================================== split billing account ===============================================
dict_path = "data/Architecture-v2/L3-Map-L1-Dict.txt"
# datapath = "data/Project-A-R-Scenario_Billing_Account-v2"
outpath = "data/Architecture-v2/v4/"
# split_bill_account(dict_path, datapath+"/Training_data.txt",
#                    outpath+"Project-A-R-BillingOnly-ReducedtoOthers/Training_data.txt",
#                    outpath+"Project-A-R-AccountOnly-ReducedtoOthers/Training_data.txt")


# ==================================== Generate 2nd layer test data ===============================================

# convert_l1_to_l3(outpath+"Project-A-R-BillingOnly-ReducedtoOthers/test_to_be_processed.txt", datapath+"/Test_data.txt",
#                  outpath+"Project-A-R-BillingOnly-ReducedtoOthers/Test_data.txt")
# convert_l1_to_l3(outpath+"/FN-sentences-layer1.txt", datapath+"/Test_data.txt",
#                  "Runs/Architecture-v2/v4/Level-1-len40-CNNv0/False-Negative-distribution-data.txt")
