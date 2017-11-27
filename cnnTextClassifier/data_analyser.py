import numpy as np
from pip.commands.list import tabulate


def analyse_data(data_path, test_path, vocab_path):
    tags = []
    tags_vocab = []

    # Load data from files
    examples = list(open(data_path, 'r', encoding="utf8").readlines())
    examples = [s.split("\t") for s in examples]
    tags.extend([s[1].strip() for s in examples])

    test = list(open(test_path,'r', encoding="utf8").readlines())
    test = [s.split("\t") for s in test]
    tags.extend(s[1].strip() for s in test)

    temp = list(open(vocab_path, 'r', encoding="utf8").readlines())
    tags_vocab.extend(s.strip() for s in temp)

    counters = np.zeros(len(tags_vocab))

    for x in tags:
        counter = 0
        for s in tags_vocab:
            if x == s:
                counters[counter] += 1
            counter += 1

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




