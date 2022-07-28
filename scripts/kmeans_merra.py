import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from yellowbrick.cluster import InterclusterDistance, silhouette_visualizer

variable_list = ["U", "V", "QV"]

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
for i in range(2, 20):
    model = KMeans(i)
    visualizer = InterclusterDistance(model)
    visualizer.fit(encodings_reduced)
    visualizer.show()

    plt.savefig('merra_clustering/intercluster_distance_%s_%d.png' % (var_string, i))
    plt.close()
    
    silhouette_visualizer(model, encodings_reduced, colors='yellowbrick')
    plt.savefig('merra_clustering/Silhouette_%s_%d.png' % (var_string, i))
    plt.close()



