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
    out_path = '/lcrc/group/earthscience/rjackson/MERRA2/hou_extended/'
    
    variable_list = ["BCCMASS", "BCFLUXU", "BCFLUXV",
        "BCSMASS", "DMSSMASS", "BCSMASS",
        "DUCMASS", "DUFLUXU", "DUFLUXV",
        "DUSMASS", "OCCMASS", "OCFLUXU", "DUSMASS25", 
        "OCFLUXV", "OCSMASS",
        "SO4CMASS", "SO4SMASS", "SSCMASS", "SSCMASS25", "SSSMASS", 
        "SSSMASS25",
        "SSFLUXU", "SSFLUXV",
        "SUFLUXU", "SUFLUXV"]

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    code = 'HOU'
    if code == 'HOU':
        ax_extent = [-100, -90, 25, 35]
    elif code == 'globe':
        ax_extent = [-120, 0, 0, 60]

    inp_ds = xr.open_mfdataset(in_merra_path, parallel=True)
    print(inp_ds)
    for variable in variable_list:
        if os.path.exists(out_path + '%s%s.nc' % (variable, sys.argv[1])):
            continue
        print("Processing %s" % variable)
        in_ds1 = inp_ds[variable].sel(
            lat=slice(ax_extent[2], ax_extent[3]), lon=slice(ax_extent[0], ax_extent[1]))
        in_ds1.to_netcdf(out_path + '%s%s.nc' % (variable, sys.argv[1]))
        in_ds1.close()
    inp_ds.close()
