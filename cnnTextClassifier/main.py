from cnnTextClassifier import train_op, eval_op
from cnnTextClassifier.config import Config
from cnnTextClassifier.text_cnn import TextCNN
from cnnTextClassifier.text_cnn_v1 import TextCNNv1
from cnnTextClassifier.text_cnn_v2 import TextCNNv2
from cnnTextClassifier.lstm_cnn import LSTMCNN


if __name__ == "__main__":
    len = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    model = {TextCNN: "CNN", LSTMCNN: "LSTM-CNN"}
    #
    # for y in model:
    #     for x in len:
    #         config = Config(enable_char=False, doc_length=x, model_name=model[y], checkpoint_dir="")
    #         train_op.train(config=config, model=y)
    #         eval_op.evaluation(config=config)

    for y in range(10):
        for a in model:
            for x in len:
                if a == LSTMCNN:
                    config = Config(run_number=y, enable_char=True, doc_length=x, model_name=model[a], checkpoint_dir="")
                else:
                    config = Config(run_number=y, enable_char=False, doc_length=x, model_name=model[a], checkpoint_dir="")
                # train_op.train(config=config, model=a)
                eval_op.evaluation(config=config)



