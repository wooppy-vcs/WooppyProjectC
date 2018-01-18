from cnnTextClassifier import train_op, eval_op, special_train_op
from cnnTextClassifier.config import Config
from cnnTextClassifier.text_cnn import TextCNN
from cnnTextClassifier.text_cnn_v1 import TextCNNv1
from cnnTextClassifier.text_cnn_v2 import TextCNNv2
from cnnTextClassifier.lstm_cnn import LSTMCNN


if __name__ == "__main__":

    version = "v4"

    model = {TextCNN: "CNNv0"}

    # dataset_name = {"Project-A-R-Level-1-ReducedtoOthers": "Level-1"}
    dataset_name = {"Project-A-R-AccountOnly-ReducedtoOthers": "Account_Only_Scenarios",
                    "Project-A-R-BillingOnly-ReducedtoOthers": "Billing_Only_Scenarios"}

    for a in model:
        # if a == LSTMCNN:
        #     config = Config(enable_char=True, transfer_learning=False, doc_length=40,
        #                     model_name=model[a], checkpoint_dir="")
        # else:
        for x in dataset_name:
            config = Config(dataset_name=x, version=version, classifier_type=dataset_name[x],
                            doc_length=40, model_name=model[a])
            train_op.train(config=config, model=a)
            # special_train_op.train(config=config, model=a)
            eval_op.evaluation(config=config)

    # config = Config(enable_char=False, transfer_learning=False, doc_length=40,
    #                 model_name='CNN', checkpoint_dir="")
    # train_op.train(config=config, model=TextCNN)
    # eval_op.evaluation(config)
