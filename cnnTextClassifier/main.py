from cnnTextClassifier import train_op, eval_op
from cnnTextClassifier import data_helpers
from cnnTextClassifier.config import Config
from cnnTextClassifier.text_cnn import TextCNN
from cnnTextClassifier.text_cnn_v1 import TextCNNv1
from cnnTextClassifier.text_cnn_v2 import TextCNNv2
from cnnTextClassifier.lstm_cnn import LSTMCNN


if __name__ == "__main__":
    # len = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # model = {TextCNN: "CNN", TextCNNv1: "CNNv1", TextCNNv2: "CNNv2"}
    #
    # for y in model:
    #     for x in len:
    #         config = Config(False, x, model_name=model[y])
    #         train_op.train(config, model=y)
    #         eval_op.evaluation(config)

    config = Config(True, 40, model_name="LSTM-CNN")
    train_op.train(config, model=LSTMCNN)

    #
    # data = []
    # config = Config()
    # datasets = data_helpers.get_datasets(data_path=config.training_path, vocab_tags_path=config.tags_vocab_path,
    #                                      config=config,
    #                                      sentences=config.data_column, tags=config.tags_column)
    # datasets_val = data_helpers.get_datasets(data_path=config.test_path, vocab_tags_path=config.tags_vocab_path,
    #                                          config=config,
    #                                          sentences=config.data_column, tags=config.tags_column)
    #
    # data.extend(datasets['data'])
    # data.extend(datasets_val['data'])
    #
    # vocab_char = data_helpers.get_char_vocab(data)
    # data_helpers.write_vocab_tags(vocab_char, config.char_vocab_path)

