import numpy as np
from pip.commands.list import tabulate


class Counter:
    def count(self, tags, tags_vocab):

        counters = np.zeros(len(tags_vocab))

        for x in tags:
            counter = 0
            for s in tags_vocab:
                if x == s:
                    counters[counter] += 1
                counter += 1
        return counters


def analyse_data(data_path, test_path, vocab_path):
    tags = []

    temp = list(open(vocab_path, 'r', encoding="utf8").readlines())
    tags_vocab = [s.strip() for s in temp]

    # Load data from files
    examples = list(open(data_path, 'r', encoding="utf8").readlines())
    examples = [s.split("\t") for s in examples]
    train_tags = [s[1].strip() for s in examples]
    tags.extend(train_tags)

    counter = Counter()

    counters_1 = counter.count(train_tags, tags_vocab)

    with open(path + "Training_data_analysis.txt", "w", encoding="utf8") as f:
        i = 0
        for tag, count in zip(tags_vocab, counters_1):
            if i != len(tags_vocab) - 1:
                f.write("{}\t{}\n".format(tag, str(int(count))))
            else:
                f.write(tag + "\t" + str(int(count)))
            i += 1
        f.close()

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

    test = list(open(test_path, 'r', encoding="utf8").readlines())
    test = [s.split("\t") for s in test]
    test_tags = [s[1].strip() for s in test]
    tags.extend(test_tags)

    counters_2 = counter.count(test_tags, tags_vocab)

    with open(path + "test_data_analysis.txt", "w", encoding="utf8") as f:
        i = 0
        for tag, count in zip(tags_vocab, counters_2):
            if i != len(tags_vocab) - 1:
                f.write("{}\t{}\n".format(tag, str(int(count))))
            else:
                f.write(tag + "\t" + str(int(count)))
            i += 1
        f.close()

    counters = counter.count(tags, tags_vocab)

    with open(path + "data_analysis.txt", "w", encoding="utf8") as f:
        i = 0
        for tag, count in zip(tags_vocab, counters):
            if i != len(tags_vocab) - 1:
                f.write("{}\t{}\n".format(tag, str(int(count))))
            else:
                f.write(tag + "\t" + str(int(count)))
            i += 1
        f.close()

path = "data/Project-A-R-Scenario-Truncated-Enriched-v2/"
analyse_data(path+"Training_data.txt", path+"Test_data.txt", path+"tags_vocab.txt")




