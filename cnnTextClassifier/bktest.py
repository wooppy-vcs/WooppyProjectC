from numpy import *
import matplotlib.pyplot as plt
from pylab import *
from sklearn import metrics


y_test = [0,0,1,5,5,5,5,4,5,4]
y_true = [0,0,1,5,5,5,5,4,5,4]

conf_arr = metrics.confusion_matrix(y_test, y_true)
print(conf_arr)
# conf_arr = [[33,2,0,0,0,0,0,0,0,1,3], [3,31,0,0,0,0,0,0,0,0,0], [0,4,41,0,0,0,0,0,0,0,1], [0,1,0,30,0,6,0,0,0,0,1], [0,0,0,0,38,10,0,0,0,0,0], [0,0,0,3,1,39,0,0,0,0,4], [0,2,2,0,4,1,31,0,0,0,2], [0,1,0,0,0,0,0,36,0,2,0], [0,0,0,0,0,0,1,5,37,5,1], [3,0,0,0,0,0,0,0,0,39,0], [0,0,0,0,0,0,0,0,0,0,38] ]

norm_conf = []
for i in conf_arr:
    a = 0
    tmp_arr = []
    a = sum(i,0)
    for j in i:
        tmp_arr.append(float(j)/float(a))
    norm_conf.append(tmp_arr)

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)
res = ax.imshow(array(norm_conf), cmap=cm.jet, interpolation='nearest')
cb = fig.colorbar(res)
plt.xticks([0,1,2,3], ['d','g','d','w'], rotation='vertical')
# ax.set_xticklabels(['d','g','d','w'])
savefig("confmat.png", format="png")