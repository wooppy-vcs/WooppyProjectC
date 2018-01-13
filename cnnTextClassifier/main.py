from cnnTextClassifier import train_op, eval_op, special_train_op
from cnnTextClassifier.config import Config
from cnnTextClassifier.text_cnn import TextCNN
from cnnTextClassifier.text_cnn_v1 import TextCNNv1
from cnnTextClassifier.text_cnn_v2 import TextCNNv2
from cnnTextClassifier.lstm_cnn import LSTMCNN


if __name__ == "__main__":

    model = {TextCNN: "CNNv0"}
        # , TextCNNv1: "CNNv1", TextCNNv2: "CNNv2"}
    # model = {TextCNNv1: "CNNv1", TextCNNv2: "CNNv2"}
    # dataset_name = {
        # "Project-A-R-Scenario_Billing_Account-v2": "Scenario",
        #             "Project-A-R-Level-1_Billing_Account-v2": "Level-1-v2"}
        #             "Project-A-R-Binary_Billing_Account-v1": "Binary"}
    # dataset_name = {"Project-A-R-Scenario-NoNone_Billing_Account-v1": "Scenario-NoNone"}
    # dataset_name = {"Project-A-R-Binary-TrainNoNone_Billing_Account-v1": "Binary-TrainNoNone"}
    # dataset_name = {"test": "test1"}
    dataset_name = {"Project-A-R-Account_Only-v1": "Account_Only_Scenarios",
                    "Project-A-R-Billing_Only-v1": "Billing_Only_Scenarios"}

    # model = {TextCNN: "CNNv0"}

    # for y in model:
    #     for x in len:
    #         config = Config(enable_char=False, doc_length=x, model_name=model[y], checkpoint_dir="")
    #         train_op.train(config=config, model=y)
    #         eval_op.evaluation(config=config)

    for a in model:
        # if a == LSTMCNN:
        #     config = Config(enable_char=True, transfer_learning=False, doc_length=40,
        #                     model_name=model[a], checkpoint_dir="")
        # else:
        for x in dataset_name:
            config = Config(dataset_name=x, classifier_type=dataset_name[x], enable_char=False, doc_length=40,
                            model_name=model[a], checkpoint_dir="")
            # train_op.train(config=config, model=a)
            # special_train_op.train(config=config, model=a)
            eval_op.evaluation(config=config)

    # config = Config(enable_char=False, transfer_learning=False, doc_length=40,
    #                 model_name='CNN', checkpoint_dir="")
    # train_op.train(config=config, model=TextCNN)
    # eval_op.evaluation(config)



