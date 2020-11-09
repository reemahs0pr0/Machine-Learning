import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN

dataset = pd.read_csv('../data/happiness_index.csv')
x = dataset.iloc[:, [1,3,4,5,6]].values

dbscan = DBSCAN(eps=0.3, min_samples=3)
clusters = dbscan.fit_predict(x)
print(clusters)

print('Outlier: ', dataset.iloc[clusters==-1, 0].values, '\n')
print('Cluster 0: ', dataset.iloc[clusters==0, 0].values, '\n')
print('Cluster 1: ', dataset.iloc[clusters==1, 0].values, '\n')
print('Cluster 2: ', dataset.iloc[clusters==2, 0].values, '\n')
print('Cluster 3: ', dataset.iloc[clusters==3, 0].values, '\n')
print('Cluster 4: ', dataset.iloc[clusters==4, 0].values)

fig = plt.figure(figsize=(7, 7))
ax = plt.axes(projection='3d')

for i in np.unique(clusters):
    if i == -1:
        label = 'Outlier'
    else:
        label = 'Cluster ' + str(i + 1)
    ax.scatter3D(xs=x[clusters==i,0],
             ys=x[clusters==i,1],
             zs=x[clusters==i,4],
            label=label)

plt.title('DBScan Clustering')
ax.set_xlabel(dataset.columns[1])
ax.set_ylabel(dataset.columns[3])
ax.set_zlabel(dataset.columns[6])
plt.legend()
plt.show()