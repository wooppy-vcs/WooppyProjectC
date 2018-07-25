#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import csv
import yaml
import matplotlib.pyplot as plt
import argparse
from cnnTextClassifier import data_helpers
from numpy import *
from pylab import *
from tensorflow.contrib import learn
from sklearn import metrics


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
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
# tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
# tf.flags.DEFINE_string("checkpoint_dir", "runs/1498842055/checkpoints", "Checkpoint directory from training run") #no04_25000
# tf.flags.DEFINE_string("checkpoint_dir", "runs/1499072698/checkpoints", "Checkpoint directory from training run") #all 25000
# tf.flags.DEFINE_string("checkpoint_dir", "runs/1499154486/checkpoints", "Checkpoint directory from training run") #no04_80000
# tf.flags.DEFINE_string("checkpoint_dir", "runs/1500262295/checkpoints", "Checkpoint directory from training run") #all 80000
tf.flags.DEFINE_string("checkpoint_dir", "runs/1507001278/checkpoints", "Checkpoint directory from training run") #all 80000
# tf.flags.DEFINE_string("checkpoint_dir", "runs/1507089882/checkpoints", "Checkpoint directory from training run") #all 80000


tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all test? data")
# tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

datasets = None
y_test = None

# CHANGE THIS: Load data. Load your own data here
dataset_name = cfg["datasets"]["default"]
print(dataset_name)
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
    elif dataset_name == "multilabel":
        datasets = data_helpers.get_datasets_multilabel(container_path=cfg["datasets"][dataset_name]["test_data_file"]["path"])
        datasets['target_names'] = cfg["datasets"][dataset_name]["categories"]

    x_raw, y_test = data_helpers.load_data_labels(datasets)
    y_test = np.argmax(y_test, axis=1)
    print("Total number of test examples: {}".format(len(y_test)))
else:
    if dataset_name == "mrpolarity":
        datasets = {"target_names": ['positive_examples', 'negative_examples']}
        x_raw = ["a masterpiece four years in the making", "everything is off."]
        y_test = [1, 0]

    elif dataset_name == "localdata":
        datasets = {"target_names": cfg["datasets"][dataset_name]["categories"]}
        print(datasets["target_names"])
        x_raw = ["Yap Jian Shen \n4, Jalan 3/50A, Taman Sri Manja, 46150, Petaling Jaya, Selangor \n+6016-652 6987 || jianshen92@gmail.com \n \nEDUCATION \nMonash University Malaysia \nBachelor of Engineering (Electrical and Computer System) \n\xe2\x80\xa2  WAM : 86.09  ||  CGPA : 3.97 || H1 First Class Honours \n\xe2\x80\xa2  Awards: JPA Scholarship (Full), Dean\xe2\x80\x99s Honour List 2013, 2014, 2015 & 2016 (Faculty of Engineering), Best Final Year \n\n July 2012 \xe2\x80\x93 July 2016 \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\nProject Award (3rd Place) \n\n \nEXPERIENCE \nEngineering Course Work \n\xe2\x80\xa2  Robotic Design  \n\xe2\x80\xa2  Analogue and Circuits Design \n\xe2\x80\xa2  Telecommunication and Image Processing \n\xe2\x80\xa2  Final Year Project (Computer Vision, Signal Processing, Mathematics) \n\nImplementing Computer Vision (Engineering) in Zebrafish Neurobehavioural Research (Medicine) \n\no \no  Automation in tracking fish\xe2\x80\x99s trajectory and analysing its behaviour \no  Develop a mathematical model that describes the fish behaviour, which can be used by neuroscientist to test and \n\ndevelop new drugs for various diseases \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n March 2014 \xe2\x80\x93 December 2014 \n\n            January 2011 \xe2\x80\x93 Present \n\n \nLeadership \nMonash University Engineering and I.T. Leadership Program 2014  \nMember \n \nRojaks Ultimate Frisbee Club Malaysia \nCo-Captain \n \nEMPLOYMENT HISTORY \nDango Digital \nWeb Developer/ Partner \n\xe2\x80\xa2  Partner of a company who specialized in building bespoke web applications. \n\xe2\x80\xa2  Strong advocator of Django, a python web framework. \n \nAstro  \nContractor \xe2\x80\x93 Web Developer \n \nCloudDesk \nIntern \n\xe2\x80\xa2  Developed a web application that display and monitor the usage of workstations (local computers) \n\xe2\x80\xa2  Assisted in a project that automates the generation of Window Server 2012 r2\xe2\x80\x99s virtual machines (VMs) through a user portal \n \nMonash University \nPeer Assisted Study Scheme(PASS) Tutor \n\xe2\x80\xa2  Mentoring program aimed at first/second year students to provide support in difficult units. (Signals Processing - Year 2) \nResearch Assistant  \n\xe2\x80\xa2  Conducted research work in a project that eventually became my Final Year Project \n\n           December 2015 \xe2\x80\x93 February 2016 \n\n                          December 2014 \xe2\x80\x93 February 2014 \n\n                        May 2017 \xe2\x80\x93 August 2017 \n\n            January 2016 \xe2\x80\x93 Present \n\n          March 2015 \xe2\x80\x93 June 2015 \n\n            \n\n            \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \nSTRATEQ Group \nProject Member   \n\xe2\x80\xa2  Remotely upgrade service pack of CPU of all Shell Petrol Stations in Malaysia \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n           July 2012 \xe2\x80\x93 August 2012  \n\n \nSKILLS \nLanguage: Fluent in English, Mandarin, Malay, and Cantonese \nTechnical Skills: MATLAB, C/C++, Microsoft Office, Visual Studio, Verilog, VHDL, Assembly, Machine Code, Eclipse(RTOS), \nQuartus, PSOC, LabView, HTML, CSS, Javascript/jQuery, PHP, MySQL, Python/Django \nPersonal Strength: Fast Learner, Team Player, Leadership, Excellent Critical Thinking Skills, Efficient \nOthers Interest & Skills: Competitive Sports (Frisbee), Literature (Writing), Music (Piano), Tutor, Competitive Video Games"]
        # y_test = [7, 1,2,6,3,7,5,6,5,8,9,10,5,11,12,6,13,6]

    elif dataset_name == "multilabel":
        datasets = {"target_names": cfg["datasets"][dataset_name]["categories"]}
        print(datasets["target_names"])
        x_raw = [
            "Yap Jian Shen \n4, Jalan 3/50A, Taman Sri Manja, 46150, Petaling Jaya, Selangor \n+6016-652 6987 || jianshen92@gmail.com \n \nEDUCATION \nMonash University Malaysia \nBachelor of Engineering (Electrical and Computer System) \n\xe2\x80\xa2  WAM : 86.09  ||  CGPA : 3.97 || H1 First Class Honours \n\xe2\x80\xa2  Awards: JPA Scholarship (Full), Dean\xe2\x80\x99s Honour List 2013, 2014, 2015 & 2016 (Faculty of Engineering), Best Final Year \n\n July 2012 \xe2\x80\x93 July 2016 \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\nProject Award (3rd Place) \n\n \nEXPERIENCE \nEngineering Course Work \n\xe2\x80\xa2  Robotic Design  \n\xe2\x80\xa2  Analogue and Circuits Design \n\xe2\x80\xa2  Telecommunication and Image Processing \n\xe2\x80\xa2  Final Year Project (Computer Vision, Signal Processing, Mathematics) \n\nImplementing Computer Vision (Engineering) in Zebrafish Neurobehavioural Research (Medicine) \n\no \no  Automation in tracking fish\xe2\x80\x99s trajectory and analysing its behaviour \no  Develop a mathematical model that describes the fish behaviour, which can be used by neuroscientist to test and \n\ndevelop new drugs for various diseases \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n March 2014 \xe2\x80\x93 December 2014 \n\n            January 2011 \xe2\x80\x93 Present \n\n \nLeadership \nMonash University Engineering and I.T. Leadership Program 2014  \nMember \n \nRojaks Ultimate Frisbee Club Malaysia \nCo-Captain \n \nEMPLOYMENT HISTORY \nDango Digital \nWeb Developer/ Partner \n\xe2\x80\xa2  Partner of a company who specialized in building bespoke web applications. \n\xe2\x80\xa2  Strong advocator of Django, a python web framework. \n \nAstro  \nContractor \xe2\x80\x93 Web Developer \n \nCloudDesk \nIntern \n\xe2\x80\xa2  Developed a web application that display and monitor the usage of workstations (local computers) \n\xe2\x80\xa2  Assisted in a project that automates the generation of Window Server 2012 r2\xe2\x80\x99s virtual machines (VMs) through a user portal \n \nMonash University \nPeer Assisted Study Scheme(PASS) Tutor \n\xe2\x80\xa2  Mentoring program aimed at first/second year students to provide support in difficult units. (Signals Processing - Year 2) \nResearch Assistant  \n\xe2\x80\xa2  Conducted research work in a project that eventually became my Final Year Project \n\n           December 2015 \xe2\x80\x93 February 2016 \n\n                          December 2014 \xe2\x80\x93 February 2014 \n\n                        May 2017 \xe2\x80\x93 August 2017 \n\n            January 2016 \xe2\x80\x93 Present \n\n          March 2015 \xe2\x80\x93 June 2015 \n\n            \n\n            \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \nSTRATEQ Group \nProject Member   \n\xe2\x80\xa2  Remotely upgrade service pack of CPU of all Shell Petrol Stations in Malaysia \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n           July 2012 \xe2\x80\x93 August 2012  \n\n \nSKILLS \nLanguage: Fluent in English, Mandarin, Malay, and Cantonese \nTechnical Skills: MATLAB, C/C++, Microsoft Office, Visual Studio, Verilog, VHDL, Assembly, Machine Code, Eclipse(RTOS), \nQuartus, PSOC, LabView, HTML, CSS, Javascript/jQuery, PHP, MySQL, Python/Django \nPersonal Strength: Fast Learner, Team Player, Leadership, Excellent Critical Thinking Skills, Efficient \nOthers Interest & Skills: Competitive Sports (Frisbee), Literature (Writing), Music (Piano), Tutor, Competitive Video Games"]

        # else:
        # datasets = {"target_names": ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']}
        # x_raw = ["The number of reported cases of gonorrhea in Colorado increased",
        #          "I am in the market for a 24-bit graphics card for a PC"]
        # y_test = [2, 1]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))
print(x_test)
# print("\nEvaluating...\n")

# Evaluation
# ==================================================
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
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        all_probabilities = None

        for idx, x_test_batch in enumerate(batches):
            print("Batch : " + str(idx))
            batch_predictions_scores = sess.run([predictions, scores], {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions_scores[0]])
            # print(batch_predictions_scores[0])
            # probabilities = softmax(batch_predictions_scores[1])

            # probabilities_op = tf.nn.softmax(batch_predictions_scores[1])
            probabilities_op = tf.nn.sigmoid(batch_predictions_scores[1])
            probabilities = sess.run(probabilities_op)
            print("probabilities:")
            print(probabilities)
            # probabilities = sigmo
            # print(batch_predictions_scores[1])
            if all_probabilities is not None:
                all_probabilities = np.concatenate([all_probabilities, probabilities])
            else:
                all_probabilities = probabilities

for idx, prediction in enumerate(all_predictions):
    print("Input       : " + x_raw[idx])
    print("Predicted   : " + datasets['target_names'][int(prediction)])
    print("")
    for i in range(0, 12):
        print(list(datasets['target_names'])[i] + " : " + str(all_probabilities[idx][i]))

print("id is")
print(datasets['id'])

print("all_probabilities is")
for probabilities in all_probabilities:
    sProbability = ""
    for probability in probabilities:
        sProbability += str(probability) + "\t"
    print(sProbability)

# Print accuracy if y_test is defined
# if y_test is not None:
#     correct_predictions = float(sum(all_predictions == y_test))
#     print("Total number of test examples: {}".format(len(y_test)))
#     print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))
#     print(datasets['target_names'])
    # print(y_test)
    # print(all_predictions)
    # available_target_names = list(datasets['target_names'])
    # available_target_names.remove('Others')
    # available_target_names.remove('Cool Gadgets')

    # for idx, i in enumerate(available_target_names):
    #     if idx not in y_test:
    #         y_test_forconf = np.append(y_test, [idx])
    #         all_predictions_forconf = np.append(all_predictions, [idx])

    # print(metrics.classification_report(y_test, all_predictions, target_names=available_target_names))
    # print(metrics.confusion_matrix(y_test, all_predictions))

    # Save the evaluation to a csv
    # result = []
    # for idx, prediction in enumerate(all_predictions):
    #     result.append("CORRECT" if y_test[idx] == prediction else "WRONG")
    #
    # # print(result)
    # readable_probabilities_array = []

    # for probabilities in all_probabilities[0]:
    #     print(probabilities)

    # for probabilities in all_probabilities:
    #     # readable_probabilities = ','.join((map(str, probabilities[0:14])))
    #     # readable_probabilities = probabilities[0:14]
    #     readable_probabilities = probabilities[0:len(available_target_names)]
    #     readable_probabilities_array.append(readable_probabilities)
    # print(readable_probabilities_array)
    # print(datasets['imageUrl'])
    # predictions_human_readable = np.column_stack((np.array(x_raw),
    #                                               np.array(datasets['imageUrl']),
    #                                               [datasets['target_names'][int(prediction)] for prediction in
    #                                                all_predictions],
    #                                               [datasets['target_names'][int(expected_label)] for expected_label in
    #                                                y_test],
    #                                               result,
    #                                               readable_probabilities_array))

    # predictions_human_readable = []
    # for idx, x in enumerate(np.array(x_raw)):
    #     predictions_human_readable.append(x + "," + datasets['target_names'][int(all_predictions[idx])] + "," +
    #                                       datasets['target_names'][int(datasets['target'][idx])] + "," +
    #                                       result[idx] + "," +
    #                                       readable_probabilities_array[idx])

    # out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
    # print("Saving evaluation to {0}".format(out_path))
    #
    # headers1 = ["Input", "ImageUrl", "Predicted", "Expected", "Accuracy"]
    # headers = headers1 + datasets['target_names']
    # with open(out_path, 'w', newline='') as f:
    #     csv.writer(f).writerow(headers)
    #     csv.writer(f).writerows(predictions_human_readable)
    #     # for item in predictions_human_readable:
    #     #     f.write("%s\n" % item)
    #     # f.close()
    # print(y_test)
    # print(shape(y_test))
    #
    # conf_arr = metrics.confusion_matrix(y_test, all_predictions)
    #
    # norm_conf = []
    # # for idx, i in enumerate(available_target_names):
    # for idx, i in enumerate(conf_arr):
    #     a = 0
    #     tmp_arr = []
    #     a = sum(i, 0)
    #     for j in i:
    #         try:
    #             tmp_arr.append(float(j) / float(a))
    #         except ZeroDivisionError:
    #             tmp_arr.append(float(0))
    #     norm_conf.append(tmp_arr)
    #
    # plt.clf()
    # print(len(available_target_names))
    # print(shape(array(norm_conf)))
    # fig = plt.figure(figsize=(18, 18))
    # ax = fig.add_subplot(111)
    # res = ax.imshow(array(norm_conf), cmap=cm.jet, interpolation='nearest')
    # cb = fig.colorbar(res)
    # plt.xticks([idx for idx, item in enumerate(available_target_names)], available_target_names, rotation='vertical')
    # plt.yticks([idx for idx, item in enumerate(available_target_names)], available_target_names, rotation='horizontal')
    # # plt.yticks(available_target_names)
    # savefig("confmat.png", format="png")
    # # savefig("confmat.pdf", format="pdf")