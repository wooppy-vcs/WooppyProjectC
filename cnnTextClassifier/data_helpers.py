import json

import numpy as np
import re
import collections

from os import listdir
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    print("data_size : " + str(data_size))
    print("num_batches_per_epoch : " + str(num_batches_per_epoch))
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def get_datasets_20newsgroup(subset='train', categories=None, shuffle=True, random_state=42):
    """
    Retrieve data from 20 newsgroups
    :param subset: train, test or all
    :param categories: List of newsgroup name
    :param shuffle: shuffle the list or not
    :param random_state: seed integer to shuffle the dataset
    :return: data and labels of the newsgroup
    """
    datasets = fetch_20newsgroups(subset=subset, categories=categories, shuffle=shuffle, random_state=random_state)

    return datasets


def get_datasets_mrpolarity(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]

    datasets = dict()
    datasets['data'] = positive_examples + negative_examples
    target = [0 for x in positive_examples] + [1 for x in negative_examples]
    datasets['target'] = target
    datasets['target_names'] = ['positive_examples', 'negative_examples']

    return datasets


def get_datasets_localdata(container_path=None, categories=None, load_content=True,
                           encoding='utf-8', shuffle=True, random_state=42):
    """
    Load text files with categories as subfolder names.
    Individual samples are assumed to be files stored a two levels folder structure.
    :param container_path: The path of the container
    :param categories: List of classes to choose, all classes are chosen by default (if empty or omitted)
    :param shuffle: shuffle the list or not
    :param random_state: seed integer to shuffle the dataset
    :return: data and labels of the dataset
    """
    datasets = load_files(container_path=container_path, categories=categories,
                          load_content=load_content, shuffle=shuffle, encoding=encoding,
                          random_state=random_state)
    # print(json.dumps(datasets))

    return datasets


def get_datasets_localdatasinglefile(data_file,categories):
    """
    # Load single tab delimited text file.
    :param container_path: The path of the container
    :param shuffle: shuffle the list or not
    :param random_state: seed integer to shuffle the dataset
    :return: data and labels of the dataset
    """
    # Load data from files
    examples = list(open(data_file, "r", encoding="utf8").readlines())
    examples = [s.split("\t") for s in examples]

    datasets = dict()
    data = [s[0].strip() for s in examples]
    imageUrl = [s[1].strip() for s in examples]
    target = [s[2].strip() for s in examples]
    # target = [s[3].strip() for s in examples] #for all data
    # target_names_dict = {0: "Others",
    #                      1: "Storage & Memory Cards",
    #                      2: "Tablets",
    #                      3: "Screen Protectors",
    #                      4: "Cool Gadgets",
    #                      5: "Cables & Charges",
    #                      6: "Mobile Phones",
    #                      7: "Cases & Covers",
    #                      8: "Mobile Car Accessories",
    #                      9: "Wearables",
    #                      10: "Audio",
    #                      11: "Powerbanks & Batteries",
    #                      12: "Camera & Accessories",
    #                      13: "Selfie Accessories"
    #                      }
    # target_names = ["Others",
    #                 "Storage & Memory Cards",
    #                 "Tablets",
    #                 "Screen Protectors",
    #                 "Cool Gadgets",
    #                 "Cables & Charges",
    #                 "Mobile Phones",
    #                 "Cases & Covers",
    #                 "Mobile Car Accessories",
    #                 "Wearables",
    #                 "Audio",
    #                 "Powerbanks & Batteries",
    #                 "Camera & Accessories",
    #                 "Selfie Accessories"
    #                 ]
    target_names = categories
    # for s in examples:
    #     target_names.append(target_names_dict[int(s[2].strip())])
    datasets['data'] = data
    datasets['target'] = [int(s) for s in target]
    datasets['target_names'] = target_names
    datasets['imageUrl'] = imageUrl

    return datasets


def get_datasets_localdatacategorizedbyfilename(container_path=None, categories=None, categories_dict=None, load_content=True,
                           encoding='utf-8', shuffle=True, random_state=42):
    """
    # Load text files categorized by filename.
    :return: data and labels of the dataset
    """
    # Load data from files

    target = []
    target_names = []
    target_names_dict = categories_dict
    data = []

    folders = [f for f in sorted(listdir(container_path))]
    print(folders)
    for label, folder in enumerate(folders):
        files = [f for f in sorted(listdir(container_path + "/" + folder))]
        foldername = folder
        # print(files)
        # print(folder)
        for sublabel, file in enumerate(files):
            # filename = file[:-4]

            with open(container_path+"/"+folder+"/"+file, 'r', encoding="utf8") as f:
                print(container_path+"/"+folder+"/"+file)
                alllines = f.readlines()
                data_element = ""
                data_count = 0
                for count, currentline in enumerate(alllines):
                    if "|##|JDNUMBER_" in currentline or count == len(alllines)-1:
                        if count != 0:
                            data_count = data_count + 1
                            # print(currentline + "\t" + str(data_count))
                            # print(str(count) + "\t" + str(len(alllines)))
                            target.append(target_names_dict[foldername])
                            data.append(data_element)
                            data_element = ""
                    else:
                        data_element = data_element + '\n' + currentline


    target_names_dict = collections.OrderedDict(sorted(categories_dict.items()))

    for k, v in target_names_dict.items():
        target_names.append(target_names_dict[k])
    print(target_names)

    datasets = dict()
    datasets['data'] = data
    datasets['target'] = target
    datasets['target_names'] = target_names
    return datasets

def load_data_labels(datasets):
    """
    Load data and labels
    :param datasets:
    :return:
    """
    # Split by words
    x_text = datasets['data']
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    labels = []
    print(len(x_text))
    print(len(datasets['target']))

    for i in range(len(x_text)):
        label = [0 for j in datasets['target_names']]
        label[datasets['target'][i]] = 1
        labels.append(label)
    y = np.array(labels)
    return [x_text, y]


def load_embedding_vectors_word2vec(vocabulary, filename, binary):
    # load embedding_vectors from the word2vec
    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initial matrix with random uniform
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        if binary:
            binary_len = np.dtype('float32').itemsize * vector_size
            count = 0
            for line_no in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':
                        word.append(ch)
                word = str(b''.join(word), encoding=encoding, errors='strict')
                word_to_check = 'iphone6s'
                if word == word_to_check:
                    print("word2vec contains : " + word_to_check)
                # print(word)
                idx = vocabulary.get(word)

                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                    count += 1
                else:
                    f.seek(binary_len, 1)

        else:
            for line_no in range(vocab_size):
                line = f.readline()
                if line == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                parts = str(line.rstrip(), encoding=encoding, errors='strict').split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                word, vector = parts[0], list(map('float32', parts[1:]))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = vector
        print("replaced " + str(count))
        print("vocab size " + str(len(vocabulary)))
        f.close()
        return embedding_vectors


def load_embedding_vectors_glove(vocabulary, filename, vector_size):
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors
