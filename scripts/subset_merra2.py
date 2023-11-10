import xarray as xr
import os 

from distributed import Client, LocalCluster

if __name__ == "__main__":
    in_path = '/lcrc/group/earthscience/rjackson/MERRA2/hou_extended/'
    out_path = '/lcrc/group/earthscience/rjackson/MERRA2/hou_subset/'

    species = ["BCSMASS", "OCSMASS", "SO4SMASS","DUSMASS",
            "SSSMASS"]

    with Client(LocalCluster(n_workers=8, threads_per_worker=1)) as c:
        print(c)
        for spec in species:
            print(spec)
            in_ds = xr.open_mfdataset(
                    os.path.join(in_path, '%s*.nc' % spec)).sel(
                            lat=slice(25, 35), lon=slice(-100, -90))
            in_ds.to_netcdf(os.path.join(out_path, '%s*.nc' % spec))
            in_ds.close()
