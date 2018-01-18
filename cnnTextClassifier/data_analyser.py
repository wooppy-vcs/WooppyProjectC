import numpy as np
from pip.commands.list import tabulate

from cnnTextClassifier import data_helpers
from cnnTextClassifier.config import Config


class Counter:
    def count(self, tags, tags_vocab):
        # tags and tags_vocab can be in digits form or words form
        counters = np.zeros(len(tags_vocab))

        for x in tags:
            counter = 0
            for s in tags_vocab:
                if x == s:
                    counters[counter] += 1
                counter += 1
        return counters


def count_and_write(data_path, outpath, tags, tags_vocab):
    examples = list(open(data_path, 'r', encoding="utf8").readlines())
    examples = [s.split("\t") for s in examples]
    train_tags = [s[1].strip() for s in examples]
    tags.extend(train_tags)

    counter = Counter()

    counters_1 = counter.count(train_tags, tags_vocab)

    with open(outpath, "w", encoding="utf8") as f:
        i = 0
        for tag, count in zip(tags_vocab, counters_1):
            if i != len(tags_vocab) - 1:
                f.write("{}\t{}\n".format(tag, str(int(count))))
            else:
                f.write(tag + "\t" + str(int(count)))
            i += 1
        f.close()
    return tags


def analyse_data(data_path, test_path, vocab_path, outpath):
    tags = []

    temp = list(open(vocab_path, 'r', encoding="utf8").readlines())
    tags_vocab = [s.strip() for s in temp]

    # Load data from files
    tags = count_and_write(data_path, outpath+"/Training_data_analysis.txt", tags, tags_vocab)

    # val = list(open(val_path, 'r', encoding="utf8").readlines())
    # val = [s.split("\t") for s in val]
    # val_tags = [s[1].strip() for s in val]
    # tags.extend(val_tags)
    #
    # counters_3 = counter.count(tags, tags_vocab)
    #
    # with open(path + "val_data_analysis.txt", "w", encoding="utf8") as f:
    #     i = 0
    #     for tag, count in zip(tags_vocab, counters_3):
    #         if i != len(tags_vocab) - 1:
    #             f.write("{}\t{}\n".format(tag, str(int(count))))
    #         else:
    #             f.write(tag + "\t" + str(int(count)))
    #         i += 1
    #     f.close()

    tags = count_and_write(test_path, outpath + "/Test_data_analysis.txt", tags, tags_vocab)

    counter = Counter()
    counters = counter.count(tags, tags_vocab)

    with open(outpath + "/Whole_data_analysis.txt", "w", encoding="utf8") as f:
        i = 0
        for tag, count in zip(tags_vocab, counters):
            if i != len(tags_vocab) - 1:
                f.write("{}\t{}\n".format(tag, str(int(count))))
            else:
                f.write(tag + "\t" + str(int(count)))
            i += 1
        f.close()

# config = Config(dataset_name="Project-A-R-Scenario_Billing_Account-v2")
# ===================================Purely for data analysing===============================================
# outpath = config.default_data_path
# analyse_data(config.training_path, config.test_path, config.tags_vocab_path, outpath)
#
# outpath = "data/Project-A-R-Scenario-Acc-Bill-Reduced-Class"
#
# # outpath = "data/Project-A-R-Level-1_Billing_Account-v2"
# analyse_data(outpath+"/Training_data.txt", outpath+"/Test_data.txt", outpath+"/tags_vocab.txt", outpath)


# ===========================To get FN distribuation for two layers architecture =====================================
# tags = []

# temp = list(open("data/Project-A-R-Scenario_Billing_Account-v2/tags_vocab.txt", 'r', encoding="utf8").readlines())
# temp = list(open("data/Project-A-R-Scenario-Acc-Bill-Reduced-Class/tags_vocab.txt", 'r', encoding="utf8").readlines())
# tags_vocab = [s.strip() for s in temp]
#
# count_and_write("Runs/Architecture-v2/v2/Level-1-len40-CNNv0/False-Negative-distribution-data.txt",
#                 "Runs/Architecture-v2/v2/Level-1-len40-CNNv0/False-Negative-distribution.txt", tags, tags_vocab)
