"""
Convert needed variables to TFRecords for each time period.
These variables are more optimally read by TensorFlow
"""

import xarray as xr
import os
import numpy as np
import sys
from glob import glob


if __name__ == "__main__":
    in_merra_path = '/lcrc/group/earthscience/rjackson/MERRA2/%s/*.nc4' % sys.argv[1]
    out_path = '/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/'
    
    variable_list = ["BCCMASS", "BCFLUXU", "BCFLUXV",
        "BCSMASS", "DMSCMASS", "DMSSMASS", 
        "DUCMASS", "DUCMASS25", "DUFLUXU", "DUFLUXV",
        "DUSMASS", "DUSMASS25", "OCCMASS", "OCFLUXU",
        "OCFLUXV", "OCSMASS", "SO2CMASS", "SO2SMASS",
        "SO4CMASS", "SO4SMASS", "SSCMASS", "SSCMASS25",
        "SSFLUXU", "SSFLUXV", "SSSMASS", "SSSMASS25",
        "SUFLUXU", "SUFLUXV"]

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    code = 'HOU'
    if code == 'HOU':
        ax_extent = [-105, -85, 25, 35]
    elif code == 'SEUS':
        ax_extent = [-90, -75, 30, 37.5]

    inp_ds = xr.open_mfdataset(in_merra_path)
    print(inp_ds)
    for variable in variable_list:
        if os.path.exists(out_path + '%s%s.nc' % (variable, sys.argv[1])):
            continue
        print("Processing %s" % variable)
        in_ds1 = inp_ds[variable]
        lon_inds = np.argwhere(
            np.logical_and(
                in_ds1.lon.values >= ax_extent[0],
                in_ds1.lon.values <= ax_extent[1])).astype(int)
        lat_inds = np.argwhere(
            np.logical_and(
                in_ds1.lat.values >= ax_extent[2],
                in_ds1.lat.values <= ax_extent[3])).astype(int)
        in_ds1 = in_ds1[:, int(lat_inds[0]):int(lat_inds[-1]), int(lon_inds[0]):int(lon_inds[-1])]
        in_ds1.load()
        in_ds1.to_netcdf(out_path + '%s%s.nc' % (variable, sys.argv[1]))
        in_ds1.close()
    inp_ds.close()
