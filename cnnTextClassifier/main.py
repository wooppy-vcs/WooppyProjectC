from cnnTextClassifier import train_op, eval_op
from cnnTextClassifier.config import Config
from cnnTextClassifier.text_cnn import TextCNN
from cnnTextClassifier.text_cnn_v1 import TextCNNv1
from cnnTextClassifier.text_cnn_v2 import TextCNNv2


if __name__ == "__main__":
    len = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    model = {TextCNN: "CNN", TextCNNv1: "CNNv1", TextCNNv2: "CNNv2"}

    for y in model:
        for x in len:
            config = Config(x, model[y])
            train_op.train(config, model=y)
            eval_op.evaluation(config)

