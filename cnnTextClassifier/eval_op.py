#! /usr/bin/env python
import numpy as np
import tensorflow as tf
import csv
from cnnTextClassifier import data_helpers
from pylab import matplotlib
matplotlib.use('Agg')
from tensorflow.contrib import learn
from sklearn import metrics
import os

from cnnTextClassifier.data_helpers import load_vocab


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))


def evaluation(config):
    # Loading training data
    # datasets = data_helpers.get_datasets(data_path=config.test_path, vocab_tags_path=config.tags_vocab_path,
    #                                      vocab_char_path=config.char_vocab_path, config=config,
    #                                      sentences=config.data_column, tags=3, tags_2=4, tags_3=5, scores=1)

    datasets = data_helpers.get_datasets(data_path=config.test_path, vocab_tags_path=config.tags_vocab_path,
                                         vocab_char_path=config.char_vocab_path, enable_char=config.enable_char,
                                         sentences=config.data_column, tags=config.tags_column)

    # Converting label to one hot vectors
    x_raw, y_test = data_helpers.load_data_labels(datasets)
    # Converting one hot vectors to indexes
    y_test = datasets['target']
    # y_test_2 = datasets['target_2']
    # y_test_3 = datasets['target_3']
    # assigned_scores = datasets['scores']
    print("=======================================================")
    print("Total number of test examples: {}".format(len(y_test)))

    # Map data into vocabulary
    model_path = config.out_dir+"/checkpoints"
    vocab_path = os.path.join(model_path, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

    # Char level processing
    if config.enable_char:
        processing_word = data_helpers.get_processing_word(vocab_chars=datasets['vocab_chars'])
        char_ids_feed = []
        for sentence in x_raw:
            words_raw = sentence.strip().split(" ")
            words = [processing_word(w) for w in words_raw]
            char_ids_feed += [words]
        char_ids_fd, word_lengths_fd = data_helpers.pad_sequences(config.doc_length, char_ids_feed, pad_tok=0, nlevels=2)
        char_ids_fd = numpy.delete(char_ids_fd, numpy.s_[config.doc_length:], 1)
        word_lengths_fd = numpy.delete(word_lengths_fd, numpy.s_[config.doc_length:], 1)

    print("=======================================================")
    print("Loading Checkpoints at {}".format(model_path))

    checkpoint_file = tf.train.latest_checkpoint(model_path)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=config.allow_soft_placement,
            log_device_placement=config.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]

            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            if config.enable_char:

                char_ids = graph.get_operation_by_name("char_ids").outputs[0]

                word_lengths = graph.get_operation_by_name("word_lengths").outputs[0]

            # Tensors we want to evaluate
            scores = graph.get_operation_by_name("output/scores").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            print("=======================================================")
            if config.enable_char:
                batches = data_helpers.batch_iter_lstm(x_test, np.zeros((x_test.shape[0], len(datasets['vocab_chars'])))
                                                       , char_ids_fd, word_lengths_fd, 1, 1,
                                                       shuffle=False)
            else:
                batches = data_helpers.batch_iter(list(x_test), 1, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            all_probabilities = None
            if config.enable_char:
                for idx, (x_test_batch, _, char_ids_batch, word_lengths_batch) in enumerate(batches):
                    # print("Batch : " + str(idx))
                    # char_ids_batch = np.expand_dims(char_ids_batch, 0)
                    # word_lengths_batch = np.expand_dims(word_lengths_batch, 0)
                    batch_predictions_scores = sess.run([predictions, scores], {input_x: x_test_batch,
                                                                                dropout_keep_prob: 1.0,
                                                                                char_ids: char_ids_batch,
                                                                                word_lengths: word_lengths_batch})
                    all_predictions = np.concatenate([all_predictions, batch_predictions_scores[0]])
                    # print(batch_predictions_scores[0])
                    probabilities = softmax(batch_predictions_scores[1])
                    # print(batch_predictions_scores[1])
                    if all_probabilities is not None:
                        all_probabilities = np.concatenate([all_probabilities, probabilities])
                    else:
                        all_probabilities = probabilities
            else:
                for idx, batch in enumerate(batches):
                    # print("Batch : " + str(idx))
                    batch_predictions_scores = sess.run([predictions, scores], {input_x: batch,
                                                                                dropout_keep_prob: 1.0})
                    all_predictions = np.concatenate([all_predictions, batch_predictions_scores[0]])
                    # print(batch_predictions_scores[0])
                    probabilities = softmax(batch_predictions_scores[1])
                    # print(batch_predictions_scores[1])
                    if all_probabilities is not None:
                        all_probabilities = np.concatenate([all_probabilities, probabilities])
                    else:
                        all_probabilities = probabilities

    print("=======================================================")
    idx_to_tag = {idx: tag for tag, idx in datasets['target_names'].items()}

    # for idx, prediction in enumerate(all_predictions):
    #     print("Input       : " + x_raw[idx])
    #     print("Predicted   : " + idx_to_tag[int(prediction)])
    #     print("")

    # Print accuracy if y_test is defined
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        # if y_test_2 or y_test_3:
        #     correct_predictions = correct_predictions + float(sum(all_predictions == y_test_2))
        #     correct_predictions = correct_predictions + float(sum(all_predictions == y_test_3))
        print("=======================================================")
        print("Total number of test examples: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))
        print("=======================================================")
        available_target_names = list(datasets['target_names'])

        y_test_forconf = y_test
        all_predictions_forconf = all_predictions.astype(int)
        for idx, i in enumerate(available_target_names):
            if idx not in y_test:
                y_test_forconf = np.append(y_test_forconf, [idx])
                all_predictions_forconf = np.append(all_predictions_forconf, [idx])

        # Calculating Precision, Recall, F score for each class
        correct_preds_tags = np.zeros(len(available_target_names))
        total_correct_tags = np.zeros(len(available_target_names))
        total_preds_tags = np.zeros(len(available_target_names))
        # corrected_scores = np.zeros(len(assigned_scores))
        k = 0

        # if y_test_2 or y_test_3:
        #     for y_test_check, y_test_2_check, y_test_3_check, y_pred, score in zip(y_test, y_test_2, y_test_3,
        #                                                                            all_predictions.astype(int), assigned_scores):
        #         if y_test_2_check and y_test_3_check:
        #             correct_preds_tags[y_test_2_check] += (int(y_test_2_check == y_pred))
        #             correct_preds_tags[y_test_3_check] += (int(y_test_3_check == y_pred))
        #             if score != 2:
        #                 corrected_scores[k] = 1
        #             else:
        #                 corrected_scores = score
        #         else:
        #             correct_preds_tags[y_test_check] += (int(y_test_check == y_pred))
        #             if y_test_2_check and y_test_3_check:
        #                 if score != 2:
        #                     corrected_scores[k] = 0
        #                 else:
        #                     corrected_scores = score
        #
        #         if y_test_2_check and y_test_3_check:
        #             total_correct_tags[y_test_2_check] += 1
        #             total_correct_tags[y_test_3_check] += 1
        #         else:
        #             total_correct_tags[y_test_check] += 1
        #
        #         total_preds_tags[y_pred] += 1
        #
        #         k += 1
        #
        # else:
        example = list(open(config.merged_map, 'r', encoding="utf8").readlines())
        f = [s.split("\t") for s in example]
        indexes = [s[0] for s in f]
        maps = [s[1].strip() for s in f]
        merged_vocab = {int(idx): int(mapped) for idx, mapped in zip(indexes, maps)}
        print(merged_vocab)
        
        y_shape = len(available_target_names)
        confusion_matrix = np.zeros([y_shape, y_shape])
        for y_test_check, y_pred in zip(y_test, all_predictions.astype(int)):
                # correct_preds_tags[y_test_check] += (int(y_test_check == y_pred))
                # total_correct_tags[y_test_check] += 1
                # total_preds_tags[y_pred] += 1
                # extra
                map1 = merged_vocab[y_test_check]
                map2 = merged_vocab[y_pred]
                confusion_matrix[map1][map2] += 1

        path = config.out_dir+"/confmat.txt"
        with open(path, 'w') as f:
            for name in available_target_names:
                f.write("\t{}".format(name))
            f.write("\n")
            for idx, name in enumerate(available_target_names):
                f.write("{}\t".format(name))
                for i in range(len(confusion_matrix[idx])):
                    f.write("{}\t".format(confusion_matrix[idx][i]))
                f.write("\n")
        # extra
        tp_by_tags = np.zeros(y_shape)
        total_predict_by_tags = np.zeros(y_shape)
        total_labeled_by_tags = np.zeros(y_shape)
        fp_by_tags = np.zeros(y_shape)
        fn_by_tags = np.zeros(y_shape)

        for idx in range(y_shape):
            tp_by_tags[idx] = confusion_matrix[idx][idx]
            for n in range(y_shape):
                total_correct_tags[idx] += confusion_matrix[idx][n]
                total_preds_tags[idx] += confusion_matrix[n][idx]

        for idx in range(y_shape):
            fp_by_tags[idx] = total_preds_tags[idx]-tp_by_tags[idx]
            fn_by_tags[idx] = total_correct_tags[idx]-tp_by_tags[idx]
        # p_tags = []
        # r_tags = []
        # f1_tags = []
        #
        # for i in range(len(available_target_names)):
        #     p_temp = correct_preds_tags[i] / total_preds_tags[i] if correct_preds_tags[i] > 0 else 0
        #     r_temp = correct_preds_tags[i] / total_correct_tags[i] if correct_preds_tags[i] > 0 else 0
        #     f1_temp = 2 * p_temp * r_temp / (p_temp + r_temp) if correct_preds_tags[i] > 0 else 0
        #     p_tags += [p_temp]
        #     r_tags += [r_temp]
        #     f1_tags += [f1_temp]
        #
        # f1_score_group = []
        # is_more_than_zero = 0
        #
        # for idx, x in enumerate(total_correct_tags):
        #     if x > 0:
        #         f1_score_group += [f1_tags[idx]]
        #         is_more_than_zero += 1

        # overall_f1 = (np.sum(f1_score_group))/is_more_than_zero

        # print("tags\tp\tr\tf1\n")
        # for idx, name in enumerate(available_target_names):
        #     print("{}\t{:04.2f}\t{:04.2f}\t{:04.2f}\n".format(name, p_tags[idx], r_tags[idx], f1_tags[idx]))
        # print("overall\t{:04.2f}\t{:04.2f}\t{:04.2f}\n\n".format(overall_p, overall_r, overall_f1))
        # print(metrics.classification_report(y_test_forconf, all_predictions_forconf, target_names=available_target_names))
        # print(metrics.classification_report(y_test, all_predictions))
        # print(metrics.confusion_matrix(y_test, all_predictions))

        # Save the evaluation to a csv
        result = []


        for idx, prediction in enumerate(all_predictions):
            # if y_test_3 or y_test_2:
            #     result.append("CORRECT" if y_test[idx] == prediction or y_test_2[idx] == prediction
            #                   or y_test_3 == prediction else "WRONG")
            # else:
            result.append("CORRECT" if y_test[idx] == prediction else "WRONG")

        readable_probabilities_array = []

        for probabilities in all_probabilities:
            readable_probabilities = probabilities[0:len(available_target_names)]
            readable_probabilities_array.append(readable_probabilities)

        tags_2 = []
        tags_3 = []

        # for expected_label in y_test_2:
        #     if expected_label != "":
        #         tags_2.append(idx_to_tag[int(expected_label)])
        #     else:
        #         tags_2.append("")
        #
        # for expected_label in y_test_3:
        #     if expected_label != "":
        #         tags_3.append(idx_to_tag[int(expected_label)])
        #     else:
        #         tags_3.append("")

        # predictions_human_readable = np.column_stack((np.array(x_raw),
        #                                               corrected_scores,
        #                                               [idx_to_tag[int(prediction)] for prediction in
        #                                                all_predictions],
        #                                               [idx_to_tag[int(expected_label)] for expected_label in
        #                                                y_test],
        #                                               tags_2,
        #                                               tags_3,
        #                                               result,
        #                                               readable_probabilities_array))

        predictions_human_readable = np.column_stack((np.array(datasets['data']),
                                                      [idx_to_tag[int(prediction)] for prediction in
                                                       all_predictions],
                                                      [idx_to_tag[int(expected_label)] for expected_label in
                                                       y_test],
                                                      result,
                                                      readable_probabilities_array))

        out_path = os.path.join(config.out_dir, "prediction-len{}.csv".format(config.doc_length))

        # Writing report
        out_path_report = os.path.join(config.out_dir, "results-len{}-confmatrix.txt".format(config.doc_length))
        with open(out_path_report, 'w', newline='') as f:
            # f.write("tags\tp\tr\tf1\tCount\n")
            f.write("tags\tTP\tFP\tFN\n")
            for idx, name in enumerate(available_target_names):
                # f.write("{}\t{:04.2f}\t{:04.2f}\t{:04.2f}\t{:04.2f}\n".format(name, p_tags[idx], r_tags[idx],
                #                                                               f1_tags[idx], total_correct_tags[idx]))
                # tp = correct_preds_tags[idx]
                tp = tp_by_tags[idx]
                # f.write("{}\t{}\t{}\t{}\n".format(name, tp, (total_preds_tags[idx]-tp), (total_correct_tags[idx]-tp)))
                f.write("{}\t{}\t{}\t{}\n".format(name, tp, (fp_by_tags[idx]), (fn_by_tags[idx])))
            # f.write("\nNo of class present\t{}\t\tOverall f1\t{:04.2f}\n".format(is_more_than_zero, overall_f1))
            # f.write(metrics.classification_report(y_test_forconf, all_predictions_forconf, target_names=available_target_names))

        f.close()

        print("=======================================================")
        print("Saving evaluation to {0}".format(out_path))
        print("=======================================================")

        # headers1 = ["Input", "Predicted", "Expected", "Expected2", "Expected3", "Accuracy"]
        headers1 = ["Input", "Predicted", "Expected", "Accuracy"]
        headers = headers1 + [tags for tags, idx in datasets['target_names'].items()]
        with open(out_path, 'w', newline='') as f:
            csv.writer(f).writerow(headers)
            csv.writer(f).writerows(predictions_human_readable)

        # Constructing a confusion matrix using sklearn
        conf_arr = metrics.confusion_matrix(y_test_forconf, all_predictions_forconf)

        norm_conf = []
        for idx, i in enumerate(conf_arr):
            a = 0
            tmp_arr = []
            a = sum(i, 0)
            for j in i:
                try:
                    tmp_arr.append(float(j) / float(a))
                except ZeroDivisionError:
                    tmp_arr.append(float(0))
            norm_conf.append(tmp_arr)

        # matplotlib.pyplot.clf()

        # fig = matplotlib.pyplot.figure(figsize=(18, 18))
        # ax = matplotlib.fig.add_subplot(111)
        # res = matplotlib.ax.imshow(array(norm_conf), cmap=cm.jet, interpolation='nearest')
        # cb = matplotlib.fig.colorbar(res)
        # matplotlib.pyplot.xticks([idx for idx, item in enumerate(available_target_names)], available_target_names, rotation='vertical')
        # matplotlib.pyplot.yticks([idx for idx, item in enumerate(available_target_names)], available_target_names, rotation='horizontal')

        # out_path = os.path.join(model_path, "..", "confmat.png")

        # savefig(out_path, format="png")
