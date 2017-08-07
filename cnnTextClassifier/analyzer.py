#! /usr/bin/env python

from numpy import *
import matplotlib.pyplot as plt
from pylab import *
from sklearn import metrics
from cnnTextClassifier import data_helpers
from tensorflow.contrib import learn
import csv
from sklearn import metrics
import yaml
import matplotlib.pylab as pylab

# data_file = "pred_expected_index_100000_excl_04_ckpt-23000.txt"
data_file = "pred_expected_index_100000_ckpt-34000.txt"
# data_file = "pred_expected_index_14000_ckpt-50000.txt"
all_data = list(open(data_file, "r").readlines())
y_test = []
y_true = []
for data in all_data:
    y_test.append(int(data.split("\t")[0]))
    y_true.append(int(data.split("\t")[1]))

print(y_test)
conf_arr = metrics.confusion_matrix(y_test, y_true)
# print(conf_arr)
# conf_arr = [[33,2,0,0,0,0,0,0,0,1,3], [3,31,0,0,0,0,0,0,0,0,0], [0,4,41,0,0,0,0,0,0,0,1], [0,1,0,30,0,6,0,0,0,0,1], [0,0,0,0,38,10,0,0,0,0,0], [0,0,0,3,1,39,0,0,0,0,4], [0,2,2,0,4,1,31,0,0,0,2], [0,1,0,0,0,0,0,36,0,2,0], [0,0,0,0,0,0,1,5,37,5,1], [3,0,0,0,0,0,0,0,0,39,0], [0,0,0,0,0,0,0,0,0,0,38] ]


with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)


dataset_name = cfg["datasets"]["default"]
datasets = data_helpers.get_datasets_localdatasinglefile(
    data_file=cfg["datasets"][dataset_name]["test_data_file"]["path"],
    categories=cfg["datasets"][dataset_name]["categories"])

available_target_names = list(datasets['target_names'])
# available_target_names.remove('Others')
# available_target_names.remove('Cool Gadgets')

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

# plt.clf()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# res = ax.imshow(array(norm_conf), cmap=cm.jet, interpolation='nearest')
# cb = fig.colorbar(res)
# plt.xticks([0,1,2,3], ['d','g','d','w'], rotation='vertical')
# # ax.set_xticklabels(['d','g','d','w'])
# savefig("confmat.png", format="png")


# params = {'legend.fontsize': 'x-large',
#           'figure.figsize': (15, 5),
#          'axes.labelsize': 'x-large',
#          'axes.titlesize':'x-large',
#          'xtick.labelsize':'x-large',
#          'ytick.labelsize':'x-large'}
# pylab.rcParams.update(params)

plt.clf()
print(len(available_target_names))
print(shape(array(norm_conf)))
fig = plt.figure(figsize=(40, 40))
ax = fig.add_subplot(111)
res = ax.imshow(array(norm_conf), cmap=cm.jet, interpolation='nearest')
cb = fig.colorbar(res)
plt.xticks([idx for idx, item in enumerate(available_target_names)], available_target_names, rotation='vertical', fontsize=20)
plt.yticks([idx for idx, item in enumerate(available_target_names)], available_target_names, rotation='horizontal', fontsize=20)
# plt.yticks(available_target_names)

figure_title = 'Image All'
# fig.suptitle('this is the figure title', fontsize=20, y=10.08)
plt.text(0.5, 1.08, figure_title,
         horizontalalignment='center',
         fontsize=40,
         transform=ax.transAxes)
plt.ylabel("Actual Class", fontsize=40)
plt.xlabel("Predicted Class", fontsize=40)
# ttl = ax.title
# ttl.set_position([.5, 1.05])


savefig("confmat_image.png", format="png")
# savefig("confmat.pdf", format="pdf")