import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

dataset = pd.read_csv('../data/happiness_index.csv')
x = dataset.iloc[:, [1,3,4,5,6]].values

kmeans = KMeans(n_clusters = 3)
clusters = kmeans.fit_predict(x)
print(clusters)

print('Cluster 0: ', dataset.iloc[clusters==0, 0].values, '\n')
print('Cluster 1: ', dataset.iloc[clusters==1, 0].values, '\n')
print('Cluster 2: ', dataset.iloc[clusters==2, 0].values)

fig = plt.figure(figsize=(7, 7))
ax = plt.axes(projection='3d')

for i in np.unique(clusters):
    ax.scatter3D(xs=x[clusters==i,0],
             ys=x[clusters==i,1],
             zs=x[clusters==i,4],
            label='Cluster ' + str(i + 1))

ax.scatter3D(xs=kmeans.cluster_centers_[:, 0], 
            ys=kmeans.cluster_centers_[:,1],
            zs=kmeans.cluster_centers_[:,4],
            s=100, c='red', label='Centroids')

plt.title('K-Means Clustering')
ax.set_xlabel(dataset.columns[1])
ax.set_ylabel(dataset.columns[3])
ax.set_zlabel(dataset.columns[6])
plt.legend()
plt.show()

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(X=x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # within cluster sum of squares
plt.show()