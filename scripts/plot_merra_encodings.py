import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from yellowbrick.cluster import InterclusterDistance, silhouette_visualizer
from matplotlib import cm

variable_list = ["U", "V", "QV"]
num_clusters = 20

encodings = []
for x in variable_list:
    encoder_ds = xr.open_dataset('3dencodings-%s.nc' % x)
    print(encoder_ds)
    encodings.append(encoder_ds["encoding"].values)
    encoder_ds.close()

encodings = np.concatenate(encodings, axis=1)
print(encodings.shape)
pc = PCA(n_components=30)
encodings_reduced = pc.fit_transform(encodings)
print(np.cumsum(pc.explained_variance_ratio_))
var_string = '_'.join(variable_list)
km = KMeans(n_clusters=num_clusters).fit(encodings_reduced)
labels = km.labels_
cmap = plt.cm.coolwarm(np.linspace(0, 1, num_clusters))
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')

for i in range(num_clusters):
    ax.scatter(encodings_reduced[labels == i,0], encodings_reduced[labels == i,1], encodings_reduced[labels == i, 2], color=cmap[i])
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
ax.view_init(-120, 30)
plt.savefig('encodings_space.png')



