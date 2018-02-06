import csv
import fileinput
import os
from collections import OrderedDict

import re

run_number = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
doc_len = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
model_type = ["CNN", "LSTM-CNN"]
runs_folder = "Enriched-x10-Runs(LSTM&CNNv0)"
csv_output = os.path.abspath(os.path.join(os.path.curdir, runs_folder))


out = []
i = 0
txt_dir = []
d = OrderedDict()
j = 0

for x in run_number:
    for y in model_type:
        for len in doc_len:
            txt_dir.append(os.path.abspath(os.path.join(os.path.curdir, runs_folder, "Runs-{}-".format(x) +
                                           "Scenario-len{}".format(len) + "-" + y + "-Enriched",
                                                        "results-len{}.txt".format(len))))
            i += 1
            if i == 10:
                with open(csv_output + "/run-{}-model-{}.csv".format(x, y), 'w', newline='') as fh_csv:
                    for line in (fileinput.input(txt_dir)):
                        j += 1
                        spl = line.strip().split("\t")
                        spl += "#"
                        d.setdefault(j, [])
                        d[j] += spl[0:]
                        if j == 80:
                            j = 0
                    for k, v in d.items():
                        v = re.sub(r"\[", "", str(v))
                        v = re.sub(r"]", "", v)
                        v = re.sub(r"'", "", v)
                        v = re.sub(r", ", "\t", v)
                        v = re.sub(r'#', "", v)
                        fh_csv.write("{}\n".format(v))
                txt_dir.clear()
                d.clear()
                i = 0

#
# with os.io.open(csv_file, 'w', newline='') as fh_csv, \
#     open(file1) as fh1, \
#     open(file2) as fh2, \
#     open(file3) as fh3:
#
#     writer = csv.writer(fh_csv, delimiter='\t')
#     writer.writerow(fieldnames)
#
#     while True:
#         out = []
#         for fh in [fh1, fh2, fh3]:
#             out.append(fh.readline().strip('\n'))
#
#         if all(out):
#             writer.writerow(out)
#         else:
#             break
