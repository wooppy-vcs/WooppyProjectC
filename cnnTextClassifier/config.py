import time

import os


class Config():

    def __init__(self, doc_length=None, model_name="", checkpoint_dir=""):

        self.runs_folder = "Enriched-runs"

        self.classifier_type = "Scenario"
        self.doc_length = doc_length
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir

        # Runs Output directories
        # self.timestamp = str(int(time.time())) + "-" + self.classifier_type
        self.out_dir = os.path.abspath(os.path.join(os.path.curdir, self.runs_folder, self.classifier_type +
                                                    "-len{}".format(doc_length) + "-" + self.model_name))

    # Data structure parameters
    data_column = 0
    tags_column = 1

    # Model Hyperparameters
    enable_word_embeddings = True
    embedding_dim = 300
    filter_sizes = [3, 4, 5]
    num_filters = [32, 64, 128]
    dropout_keep_prob = 0.5
    l2_reg_lambda = 0.5

    # Early Stopping Criterion
    patience = 16
    min_delta = 0.01

    # Training parameters
    batch_size = 64
    num_epochs = 50
    num_checkpoints = 5

    # Misc Parameters
    allow_soft_placement = True
    log_device_placement = False

    # Documents Directories
    default_raw_data_path = "data/Project-A-R-Tagged-Data-Truncated-Enriched"
    dataset_name = "Project-A-R-Scenario-Truncated-Enriched-v2"
    default_data_path = "data/"+dataset_name

    training_path = default_data_path + "/Training_data.txt"
    test_path = default_data_path + "/Test_data.txt"
    tags_vocab_path = default_data_path + "/tags_vocab.txt"

    # Word Embeddings
    embedding_name = "word2vec"
    # embedding_name = "glove"

    word2vec = True
    word2vec_path = "data/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin"

    glove_path = None
    glove_dim = None

    # Char Embeddings
    dim_char = 100
    char_hidden_size = 100

