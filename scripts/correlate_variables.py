import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cartopy.crs as ccrs
import sys

from sklearn import preprocessing
from sklearn.decomposition import PCA
from distributed import LocalCluster, Client

if __name__ == '__main__':
    nc_path = '/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/'

    code = 'HOU'
    if code == 'HOU':
        ax_extent = [-105, -85, 25, 35]
    elif code == 'SEUS':
        ax_extent = [-90, -75, 30, 37.5]

    var1 = sys.argv[1]
    var2 = sys.argv[2]

    var1 = xr.open_mfdataset(nc_path + '%s*.nc' % var1)[var1].values.flatten()
    var2 = xr.open_mfdataset(nc_path + '%s*.nc' % var2)[var2].values.flatten()

    fig = plt.figure(figsize=(8, 5))
    ax = plt.axes()
    ax.scatter(var1, var2)
    corr = np.corrcoef(var1, var2)[0, 1]
    fit = np.polyfit(var1, var2, 1)
    x = np.linspace(var1.min(), var1.max(), 10)
    ax.plot(x, np.polyval(fit, x), linewidth=2, color='k')
    ax.set_xlabel(sys.argv[1])
    ax.set_ylabel(sys.argv[2])
    ax.set_title('Correlation = %3f' % corr)
    fig.savefig('ScatPlot%s%s.png' % (sys.argv[1], sys.argv[2]))
    plt.close(fig)

    
