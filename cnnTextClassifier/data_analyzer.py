

sentences=0
tags=1

data_path = "data/Project-A-R/Training_data.txt"

examples = list(open(data_path, 'r', encoding="utf8").readlines())
examples = [s.split("\t") for s in examples]
data = [s[sentences].strip() for s in examples]
target_names = [s[tags].strip() for s in examples]

counter_None = 0
counter_Wrong = 0
counter_Bill = 0

for s in target_names:
    if s == "None":
        counter_None += 1
    if s == "Wrong":
        counter_Wrong += 1
    if s == "Billing":
        counter_Bill += 1


print("Total None Tags : {}".format(counter_None))
print("Total Wrong Tags : {}".format(counter_Wrong))
print("Total Bill Tags : {}".format(counter_Bill))


