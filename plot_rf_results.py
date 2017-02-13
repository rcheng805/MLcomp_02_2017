import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

data = np.genfromtxt('crossVal_Pipeline_rf_rf_Results')

ind_1 = data[:, 0] == 30
ind_2 = data[:, 1] == 128
final_ind = (ind_1 & ind_2) > 0  #range(len(ind_1)) #

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[final_ind, 2], data[final_ind, 3], data[final_ind, 4])
ax.scatter(data[final_ind, 2], data[final_ind, 3], data[final_ind, 5], color='red')
ax.set_xlabel('Number of estimators')
ax.set_ylabel('Minimum leaf size')
ax.set_zlabel('Classification accuracy')
plt.show()
plt.close()

plt.scatter(data[data[:, 1] == 128, 0], data[data[:, 1] == 128, 4])
plt.scatter(data[data[:, 1] == 128, 0]+5, data[data[:, 1] == 128, 5], color='red')
plt.ylabel('Classification accuracy')
plt.xlabel('Number of estimators')
plt.show()
