import xarray as xr
import pickle
import glob
import os
import numpy as np

rels = glob.glob('relevance_pickles/*.pickle')
rel_out = '/lcrc/group/earthscience/rjackson/opencrums/scripts/relevance_nc/'

code = 'HOU'
if code == 'HOU':
    ax_extent = [-120, -70, 5, 55]

ds = xr.open_mfdataset(
            '/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/DUCMASS2010.nc')
print(ds.time)
lon = ds["lon"].values
lat = ds["lat"].values
lon_inds = np.argwhere(
            np.logical_and(
                ds.lon.values >= ax_extent[0],
                ds.lon.values <= ax_extent[1])).astype(int)
lat_inds = np.argwhere(
            np.logical_and(
                ds.lat.values >= ax_extent[2],
                ds.lat.values <= ax_extent[3])).astype(int)
lon = lon[lon_inds]
lat = lat[lat_inds]

#lon, lat = np.meshgrid(lon, lat)
for i, rel_file in enumerate(rels):
    out_ds = {}
    means = []
    fi = open(rel_file, mode='rb')
    pick = pickle.load(fi)
    if i % 100 == 0:
        print('%d/%d' % (i, len(rels)))

    for key in pick.keys():
        if "input_" in key:
            arr = pick[key].numpy()
            times = pick["time"]
            ind_sort = np.argsort(times)
            arr = arr[ind_sort, :, :, :]
            times = times[ind_sort]
            out_ds["rel_" + key[6:]] = xr.DataArray(
                    np.squeeze(arr), dims=('time', 'lat', 'lon'))
            means.append(np.nanmean(arr))
    means = np.mean(np.array(means))

    for key in pick.keys():
        if "input_" in key:
            out_ds["rel_" + key[6:]] /= means

    out_ds['time'] = times[ind_sort]
    out_ds['label'] = ('time', pick['output'].numpy().argmax(axis=1)[ind_sort])
    out_ds['lat'] = ('lat', np.squeeze(lat))
    out_ds['lon'] = ('lon', np.squeeze(lon))
    out_ds = xr.Dataset(out_ds)
    out_ds.to_netcdf(os.path.join(rel_out, 'relevance%d.nc' % i))
    del out_ds

