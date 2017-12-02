#! /usr/bin/env python

import tensorflow as tf
import csv
import yaml
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


with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

# Parameters
# ==================================================

# Data Parameters

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 64)")



## RUNS
#  tf.flags.DEFINE_string("checkpoint_dir", "runs/1510746346/checkpoints", "Checkpoint directory from training run") #no penalised
# tf.flags.DEFINE_string("checkpoint_dir", "runs/1510799220/checkpoints", "Checkpoint directory from training run") #penalised(10epochs)
# tf.flags.DEFINE_string("checkpoint_dir", "runs/1510744440/checkpoints", "Checkpoint directory from training run") #penalised(100epochs)
# tf.flags.DEFINE_string("checkpoint_dir", "runs/1510820356/checkpoints", "Checkpoint directory from training run") #penalised(10epochs)Binary
# tf.flags.DEFINE_string("checkpoint_dir", "runs/1510889604/checkpoints", "Checkpoint directory from training run") #penalised(10epochs)Binary(different weight)
# tf.flags.DEFINE_string("checkpoint_dir", "runs/1510910997-Level-2/checkpoints", "Checkpoint directory from training run") #penalised(10epochs)Binary(different weight)
# tf.flags.DEFINE_string("checkpoint_dir", "runs/1511176830-Scenario-len20-correctedweightedaccuracy-filtersize345-enrich/checkpoints", "Checkpoint directory from training run")
# tf.flags.DEFINE_string("checkpoint_dir", "runs/1511240087-AnswerType-len20-correctedweightedaccuracy-filtersize345-enrich/checkpoints", "Checkpoint directory from training run")
# tf.flags.DEFINE_string("checkpoint_dir", "runs/1511260117-Scenario-len20-correctedweightedaccuracy-filtersize345-remove-none/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_string("checkpoint_dir", "Enriched-runs/Scenario-len10-CNN/checkpoints", "Checkpoint directory from training run")


tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")
tf.flags.DEFINE_integer("sentences_column", 0, "Column number of sentence data in data txt file")
tf.flags.DEFINE_integer("tags_column", 1, "Column number of tags in data txt file")

# tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")
# tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
# tf.flags.DEFINE_string("checkpoint_dir", "runs/1498842055/checkpoints", "Checkpoint directory from training run") #no04_25000
# tf.flags.DEFINE_string("checkpoint_dir", "runs/1499072698/checkpoints", "Checkpoint directory from training run") #all 25000
# tf.flags.DEFINE_string("checkpoint_dir", "runs/1499154486/checkpoints", "Checkpoint directory from training run") #no04_80000

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("=======================================================")
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
print("")

datasets = None
y_test = None

# CHANGE THIS: Load data. Load your own data here
dataset_name = cfg["datasets"]["default"]

# print(dataset_name)
if FLAGS.eval_train:
    if dataset_name == "mrpolarity":
        datasets = data_helpers.get_datasets_mrpolarity(cfg["datasets"][dataset_name]["positive_data_file"]["path"],
                                                        cfg["datasets"][dataset_name]["negative_data_file"]["path"])
    elif dataset_name == "20newsgroup":
        datasets = data_helpers.get_datasets_20newsgroup(subset="test",
                                                         categories=cfg["datasets"][dataset_name]["categories"],
                                                         shuffle=cfg["datasets"][dataset_name]["shuffle"],
                                                         random_state=cfg["datasets"][dataset_name]["random_state"])
    elif dataset_name == "localdata":
        datasets = data_helpers.get_datasets_localdata(container_path=cfg["datasets"][dataset_name]["test_data_file"]["path"])

    elif dataset_name == "localdatasingledata":
        datasets = data_helpers.get_datasets_localdatasinglefile(data_file=cfg["datasets"][dataset_name]["test_data_file"]["path"],
                                                                 categories=cfg["datasets"][dataset_name]["categories"])
    elif dataset_name == "localfile":
        datasets = data_helpers.get_datasets(data_path=cfg["datasets"][dataset_name]["test_data_file"]["path"],
                                             vocab_tags_path=cfg["datasets"][dataset_name]["vocab_write_path"]["path"],
                                             sentences=FLAGS.sentences_column, tags=FLAGS.tags_column)

    x_raw, y_test = data_helpers.load_data_labels(datasets)

    y_test = np.argmax(y_test, axis=1)
    print("=======================================================")
    print("Total number of test examples: {}".format(len(y_test)))
else:
    if dataset_name == "mrpolarity":
        datasets = {"target_names": ['positive_examples', 'negative_examples']}
        x_raw = ["a masterpiece four years in the making", "everything is off."]
        y_test = [1, 0]

    elif dataset_name == "localdatasingledata":
        datasets = {"target_names": cfg["datasets"][dataset_name]["categories"]}
        print(datasets["target_names"])
        # x_raw = ["ipadmini", "ipad mini", "iphone5s", "s7edge", "A7-10"]
        x_raw = ["What is my payment status?", "Check bill", "cannot pay bill", "line barred how?"]
        # y_test = [7, 1,2,6,3,7,5,6,5,8,9,10,5,11,12,6,13,6]

    # else:
        # datasets = {"target_names": ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']}
        # x_raw = ["The number of reported cases of gonorrhea in Colorado increased",
        #          "I am in the market for a 24-bit graphics card for a PC"]
        # y_test = [2, 1]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))


# print(x_test)
# print("\nEvaluating...\n")

# Evaluation
# ==================================================
print("=======================================================")
print(FLAGS.checkpoint_dir)
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        scores = graph.get_operation_by_name("output/scores").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]


        # Generate batches for one epoch
        print("=======================================================")
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        all_probabilities = None

        for idx, x_test_batch in enumerate(batches):
            print("Batch : " + str(idx))
            batch_predictions_scores = sess.run([predictions, scores], {input_x: x_test_batch, dropout_keep_prob: 1.0})
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
    # a = x_raw[idx]
    # a1 = idx_to_tag[int(prediction)]
    print("Predicted   : " + idx_to_tag[int(prediction)])
    print("")

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("=======================================================")
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))
    print("=======================================================")
    # print(datasets['target_names'])
    # print(y_test)
    available_target_names = list(datasets['target_names'])
    # print(all_predictions)
    # available_target_names.remove('Others')
    # available_target_names.remove('Cool Gadgets')

    y_test_forconf = y_test
    all_predictions_forconf = all_predictions.astype(int)
    for idx, i in enumerate(available_target_names):
        if idx not in y_test:
            y_test_forconf = np.append(y_test_forconf, [idx])
            all_predictions_forconf = np.append(all_predictions_forconf, [idx])

    correct_preds_tags, total_correct_tags, total_preds_tags = np.zeros(len(available_target_names)), np.zeros(len(available_target_names)), np.zeros(len(available_target_names))

    for y_test_check, y_pred in zip(y_test, all_predictions.astype(int)):
        correct_preds_tags[y_test_check] += (int(y_test_check == y_pred))
        total_correct_tags[y_test_check] += 1
        total_preds_tags[y_pred] += 1

    # correct_preds_tags = [x * y for x, y in zip(correct_preds_tags, weightsArray)]
    # total_correct_tags = [x * y for x, y in zip(total_correct_tags, weightsArray)]
    # total_preds_tags = [x * y for x, y in zip(total_preds_tags, weightsArray)]

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

    out_path_report = os.path.join(FLAGS.checkpoint_dir, "..", "results.txt")
    with open(out_path_report, 'w', newline='') as f:
        f.write("tags\tp\tr\tf1\n")
        for idx, name in enumerate(available_target_names):
            f.write("{}\t{:04.2f}\t{:04.2f}\t{:04.2f}\n".format(name, p_tags[idx], r_tags[idx], f1_tags[idx]))
        f.write("overall\t{:04.2f}\t{:04.2f}\t{:04.2f}\n\n".format(overall_p, overall_r, overall_f1))
        f.write(metrics.classification_report(y_test_forconf, all_predictions_forconf, target_names=available_target_names))

    f.close()

    print("tags\tp\tr\tf1\n")
    for idx, name in enumerate(available_target_names):
        print("{}\t{:04.2f}\t{:04.2f}\t{:04.2f}\n".format(name, p_tags[idx], r_tags[idx], f1_tags[idx]))
    print("overall\t{:04.2f}\t{:04.2f}\t{:04.2f}\n\n".format(overall_p, overall_r, overall_f1))
    print(metrics.classification_report(y_test_forconf, all_predictions_forconf, target_names=available_target_names))
    # print(metrics.classification_report(y_test, all_predictions))
    # print(metrics.confusion_matrix(y_test, all_predictions))

    # Save the evaluation to a csv
    result = []
    for idx, prediction in enumerate(all_predictions):
        result.append("CORRECT" if y_test[idx] == prediction else "WRONG")

    # print(result)
    readable_probabilities_array = []

    # for probabilities in all_probabilities[0]:
    #     print(probabilities)

    for probabilities in all_probabilities:
        # readable_probabilities = ','.join((map(str, probabilities[0:14])))
        # readable_probabilities = probabilities[0:14]
        readable_probabilities = probabilities[0:len(available_target_names)]
        readable_probabilities_array.append(readable_probabilities)
    # print(readable_probabilities_array)
    # print(datasets['imageUrl'])
    predictions_human_readable = np.column_stack((np.array(x_raw),
                                                  [idx_to_tag[int(prediction)] for prediction in
                                                   all_predictions],
                                                  [idx_to_tag[int(expected_label)] for expected_label in
                                                   y_test],
                                                  result,
                                                  readable_probabilities_array))

    # predictions_human_readable = []
    # for idx, x in enumerate(np.array(x_raw)):
    #     predictions_human_readable.append(x + "," + datasets['target_names'][int(all_predictions[idx])] + "," +
    #                                       datasets['target_names'][int(datasets['target'][idx])] + "," +
    #                                       result[idx] + "," +
    #                                       readable_probabilities_array[idx])

    out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
    print("=======================================================")
    print("Saving evaluation to {0}".format(out_path))
    print("=======================================================")
    headers1 = ["Input", "Predicted", "Expected", "Accuracy"]
    headers = headers1 + [tags for tags, idx in datasets['target_names'].items()]
    with open(out_path, 'w', newline='') as f:
        csv.writer(f).writerow(headers)
        csv.writer(f).writerows(predictions_human_readable)
        # for item in predictions_human_readable:
        #     f.write("%s\n" % item)
        # f.close()
    # print(y_test)
    # print(shape(y_test))

    # conf_arr = metrics.confusion_matrix(y_test, all_predictions)
    conf_arr = metrics.confusion_matrix(y_test_forconf, all_predictions_forconf)

    norm_conf = []
    # for idx, i in enumerate(available_target_names):
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
    # print(len(available_target_names))
    # print(shape(array(norm_conf)))
    fig = plt.figure(figsize=(18, 18))
    ax = fig.add_subplot(111)
    res = ax.imshow(array(norm_conf), cmap=cm.jet, interpolation='nearest')
    cb = fig.colorbar(res)
    plt.xticks([idx for idx, item in enumerate(available_target_names)], available_target_names, rotation='vertical')
    plt.yticks([idx for idx, item in enumerate(available_target_names)], available_target_names, rotation='horizontal')
    # plt.yticks(available_target_names)
    out_path = os.path.join(FLAGS.checkpoint_dir, "..", "confmat.png")

    savefig(out_path, format="png")
    # savefig("confmat.pdf", format="pdf")