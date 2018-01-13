import time

import os


class Config():

    def __init__(self, dataset_name, classifier_type="", run_number=None, doc_length=None, model_name="",
                 checkpoint_dir="", transfer_learning=False, enable_char=False):

        self.runs_folder = "Runs/Account_Billing_v2"
        self.classifier_type = classifier_type
        self.doc_length = doc_length
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir

        self.enable_char = enable_char
        self.transfer_learning = transfer_learning

        # Runs Output directories
        # self.timestamp = str(int(time.time())) + "-" + self.classifier_type
        # self.out_dir = os.path.abspath(os.path.join(os.path.curdir, self.runs_folder, "runs-{}-".format(run_number) +
        #                                             self.classifier_type + "-len{}".format(doc_length) + "-" +
        #                                             self.model_name))
        self.out_dir = os.path.abspath(os.path.join(os.path.curdir, self.runs_folder, self.classifier_type +
                                                    "-len{}".format(doc_length) + "-" +
                                                    self.model_name))

        self.dataset_name = dataset_name
        default_data_path = "data/" + self.dataset_name

        self.training_path = default_data_path + "/Training_data.txt"
        self.test_path = default_data_path + "/Test_data.txt"
        self.development_path = default_data_path + "/Development_data.txt"
        self.tags_vocab_path = default_data_path + "/tags_vocab.txt"
        self.char_vocab_path = default_data_path + "/chars_vocab.txt"

    # Data structure parameters
    data_column = 0
    tags_column = 1

    dev_sample_fraction = 0.2

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
    default_raw_data_path = "raw_data/Project-A-R-v2.0_Billing_Account"
    # dataset_name = "Project-A-R-Scenario_Billing_Account-v2"

    # dataset_name = "Project-A-R-Scenario-Truncated-Enriched-v2"


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

    # Transfer Learning
    # pretrained_model_path = 'transfer_learning_models/1112/saved_model.pb'
    pretrained_model_path = os.path.abspath(os.path.join(os.path.curdir, 'transfer_learning_models/1412/model.weights'))
    # pretrained_weights = 'transfer_learning_models/1412/model.weights/.data-00000-of-00001'


