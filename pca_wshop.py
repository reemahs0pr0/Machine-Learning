import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('../data/wine.csv')

print(dataset)

x = dataset.iloc[:,1:].values
print(x)

# get labels
y = dataset.iloc[:,0].values
print(y)

scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components=6)
pc = pca.fit_transform(x)

print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())

fig = plt.figure(figsize=(7, 7))
ax = plt.axes(projection='3d')

for label in np.unique(y):
	pc_by_label = pc[y==label]
	ax.scatter3D(pc_by_label[:,0], pc_by_label[:,1],
    	pc_by_label[:,2])

ax.set_xlabel('Principal Compnent 1')
ax.set_ylabel('Principal Compnent 2')
ax.set_zlabel('Principal Compnent 3')
plt.title('PCA on wine dataset')
plt.legend()
plt.show()