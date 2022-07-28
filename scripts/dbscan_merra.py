import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from yellowbrick.cluster import InterclusterDistance, silhouette_visualizer

variable_list = ["OMEGA"]

encodings = []
for x in variable_list:
    encoder_ds = xr.open_dataset('3dencodings-%s.nc' % x)
    print(encoder_ds)
    encodings.append(encoder_ds["encoding"].values)
    encoder_ds.close()

encodings = np.concatenate(encodings, axis=1)
print(encodings.shape)
pc = PCA(n_components=10)
encodings_reduced = pc.fit_transform(encodings)

var_string = '_'.join(variable_list)

eps_array = np.linspace(0.2, 0.4, 10)
n_feats = np.zeros_like(eps_array)

neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(encodings_reduced)
distances, indices = nbrs.kneighbors(encodings_reduced)
distances = np.sort(distances, axis=0)
distances = distances[:,1]

plt.figure(figsize=(10, 5))
plt.plot(distances)
plt.ylabel("distance")
plt.savefig('DBSCAN_%s.png' % var_string)
plt.close()

model = DBSCAN(eps=0.1, min_samples=12).fit(encodings_reduced)
plt.figure(figsize=(10, 10))
labels = model.labels_
for i in np.unique(labels):
    inds = np.argwhere(labels == i)
    plt.scatter(encodings_reduced[inds, 0], encodings_reduced[inds, 1], label='%d' % i)
plt.legend()
plt.savefig('Cluster_plot.png')
plt.xlabel("PC 1")
plt.ylabel("PC 2")

plt.figure(figsize=(10, 10))
label_hist, bins = np.histogram(labels, bins=np.arange(-1.5, np.max(labels)+0.5, 1))
plt.step(bins[:-1], label_hist)
plt.savefig('DBSCAN_label_dist.png')

