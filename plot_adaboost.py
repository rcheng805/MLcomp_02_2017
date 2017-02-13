import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv

data = []
with open('results_adaboost_1_clean.csv', 'r') as ff:
    f = csv.reader(ff)
    for row in f:
        data.append([float(item.split('=')[1]) for item in row])

data = np.array(data)

ind_1 = data[:, 0] == 70
ind_2 = data[:, 2] == 16
final_ind = (ind_1 & ind_2) > 0 #range(len(ind_1)) #

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[final_ind, 1], data[final_ind, 3], data[final_ind, 4])
ax.set_xlabel('Learning rate')
ax.set_ylabel('Number of estimators')
ax.set_zlabel('Classification accuracy')
plt.show()
plt.close()

# plt.scatter(data[data[:, 1] == 128, 0], data[data[:, 1] == 128, 4])
# plt.scatter(data[data[:, 1] == 128, 0]+5, data[data[:, 1] == 128, 5], color='red')
# plt.ylabel('Classification accuracy')
# plt.xlabel('Number of estimators')
# plt.show()
