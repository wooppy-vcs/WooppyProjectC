

class REST_Config():

    default_dir = "/home/wooppy/Projects/BongTest/WooppyProjectC/cnnTextClassifier/"
    default_run_dir = default_dir+"Runs/Architecture-v2/v1/"
    default_data_dir = default_dir+"data/Architecture-v2/v1/"

    l1_checkpoint_dir = default_run_dir+"Level-1-v2-len40-CNNv0/checkpoints"
    billing_checkpoint_dir = default_run_dir+"Billing_Only_Scenarios-len40-CNNv0/checkpoints"
    account_checkpoint_dir = default_run_dir+"Account_Only_Scenarios-len40-CNNv0/checkpoints"

    l1_tags_vocab = default_data_dir+"Project-A-R-Level-1_Billing_Account-v2/tags_vocab.txt"
    billing_tags_vocab = default_data_dir+"Project-A-R-Billing_Only-v1/tags_vocab_main.txt"
    account_tags_vocab = default_data_dir+"Project-A-R-Account_Only-v1/tags_vocab_main.txt"

    # # for logging
    # layer_1_dir = "cnnTextClassifier/Runs/Architecture-v2/v1/Level-1-v2-len40-CNNv0/checkpoints/model-27354.meta"
    # layer_2_billing_dir = "cnnTextClassifier/Runs/Architecture-v2/v1/Billing_Only_Scenarios-len40-CNNv0/checkpoints/model-2058.meta"
    # layer_2_account_dir = "cnnTextClassifier/Runs/Architecture-v2/v1/Account_Only_Scenarios-len40-CNNv0/checkpoints/model-4150.meta"
