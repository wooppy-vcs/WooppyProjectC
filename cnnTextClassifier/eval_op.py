#! /usr/bin/env python

import tensorflow as tf
import csv
from cnnTextClassifier import data_helpers
from pylab import *
from tensorflow.contrib import learn
from sklearn import metrics
import os


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))


def evaluation(config):
    # Loading training data
    datasets = data_helpers.get_datasets(data_path=config.test_path, vocab_tags_path=config.tags_vocab_path,
                                         sentences=config.data_column, tags=config.tags_column)

    # Converting label to one hot vectors
    x_raw, y_test = data_helpers.load_data_labels(datasets)
    # Converting one hot vectors to indexes
    y_test = np.argmax(y_test, axis=1)

    print("=======================================================")
    print("Total number of test examples: {}".format(len(y_test)))

    # Map data into vocabulary
    model_path = config.out_dir+"/checkpoints"
    vocab_path = os.path.join(model_path, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

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

            # Tensors we want to evaluate
            scores = graph.get_operation_by_name("output/scores").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            print("=======================================================")
            batches = data_helpers.batch_iter(list(x_test), 1, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            all_probabilities = None

            for idx, x_test_batch in enumerate(batches):
                print("Batch : " + str(idx))
                batch_predictions_scores = sess.run([predictions, scores], {input_x: x_test_batch,
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

    for idx, prediction in enumerate(all_predictions):
        print("Input       : " + x_raw[idx])
        print("Predicted   : " + idx_to_tag[int(prediction)])
        print("")

    # Print accuracy if y_test is defined
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
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

        for y_test_check, y_pred in zip(y_test, all_predictions.astype(int)):
            correct_preds_tags[y_test_check] += (int(y_test_check == y_pred))
            total_correct_tags[y_test_check] += 1
            total_preds_tags[y_pred] += 1

        p_tags = []
        r_tags = []
        f1_tags = []

        for i in range(len(available_target_names)):
            p_temp = correct_preds_tags[i] / total_preds_tags[i] if correct_preds_tags[i] > 0 else 0
            r_temp = correct_preds_tags[i] / total_correct_tags[i] if correct_preds_tags[i] > 0 else 0
            f1_temp = 2 * p_temp * r_temp / (p_temp + r_temp) if correct_preds_tags[i] > 0 else 0
            p_tags += [p_temp]
            r_tags += [r_temp]
            f1_tags += [f1_temp]

        overall_p = sum(correct_preds_tags)/sum(total_preds_tags)
        overall_r = sum(correct_preds_tags)/sum(total_correct_tags)
        overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r)

        # Writing report
        out_path_report = os.path.join(model_path, "..", "results.txt")
        with open(out_path_report, 'w', newline='') as f:
            f.write("tags\tp\tr\tf1\n")
            for idx, name in enumerate(available_target_names):
                f.write("{}\t{:04.2f}\t{:04.2f}\t{:04.2f}\n".format(name, p_tags[idx], r_tags[idx], f1_tags[idx]))
            f.write("overall\t{:04.2f}\t{:04.2f}\t{:04.2f}\n\n".format(overall_p, overall_r, overall_f1))
            f.write(metrics.classification_report(y_test_forconf, all_predictions_forconf, target_names=available_target_names))

        f.close()

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
            result.append("CORRECT" if y_test[idx] == prediction else "WRONG")

        readable_probabilities_array = []

        for probabilities in all_probabilities:
            readable_probabilities = probabilities[0:len(available_target_names)]
            readable_probabilities_array.append(readable_probabilities)

        predictions_human_readable = np.column_stack((np.array(x_raw),
                                                      [idx_to_tag[int(prediction)] for prediction in
                                                       all_predictions],
                                                      [idx_to_tag[int(expected_label)] for expected_label in
                                                       y_test],
                                                      result,
                                                      readable_probabilities_array))

        out_path = os.path.join(model_path, "..", "prediction.csv")

        print("=======================================================")
        print("Saving evaluation to {0}".format(out_path))
        print("=======================================================")

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

        plt.clf()

        fig = plt.figure(figsize=(18, 18))
        ax = fig.add_subplot(111)
        res = ax.imshow(array(norm_conf), cmap=cm.jet, interpolation='nearest')
        cb = fig.colorbar(res)
        plt.xticks([idx for idx, item in enumerate(available_target_names)], available_target_names, rotation='vertical')
        plt.yticks([idx for idx, item in enumerate(available_target_names)], available_target_names, rotation='horizontal')

        out_path = os.path.join(model_path, "..", "confmat.png")

        savefig(out_path, format="png")
