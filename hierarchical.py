import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# hierarchical clustering of iris
dataset = pd.read_csv('../data/iris.csv')
x = dataset.iloc[:, 1:-1].values

from scipy.cluster.hierarchy import dendrogram, linkage

plt.figure(figsize=(25, 10))
plt.title('Iris Hierarchical Clustering Dendrogram')
plt.xlabel('Species')
plt.ylabel('distance')

dendrogram(
    linkage(x, 'ward'),  # generate the linkage matrix
    leaf_font_size=8      # font size for the x axis labels
)

plt.axhline(y=8)
plt.show()


# plot 2d graphs
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(linkage="ward", n_clusters=3)
clustering.fit(x)

print(clustering.labels_)

for i in np.unique(clustering.labels_):
    plt.scatter(x=x[clustering.labels_==i, 0], 
    			y=x[clustering.labels_==i, 1],
                label='Cluster ' + str(i + 1))

plt.title('Hierarchical Clustering')
plt.xlabel(dataset.columns[1])
plt.ylabel(dataset.columns[2])
plt.legend()
plt.show()