import matplotlib.pyplot as plt
import xarray as xr
import os
import numpy as np


variable_list = ["BCCMASS", "BCFLUXU", "BCFLUXV",
        "BCSMASS", "DMSCMASS", "DMSSMASS",
        "DUCMASS", "DUCMASS25", "DUFLUXU", "DUFLUXV",
        "DUSMASS", "DUSMASS25", "OCCMASS", "OCFLUXU",
        "OCFLUXV", "OCSMASS", "SO2CMASS", "SO2SMASS",
        "SO4CMASS", "SO4SMASS", "SSCMASS", "SSCMASS25",
        "SSFLUXU", "SSFLUXV", "SSSMASS", "SSSMASS25",
        "SUFLUXU", "SUFLUXV"]

merra_eof_path = '/lcrc/group/earthscience/rjackson/opencrums/models/EOFs/'

num_eofs = {}
for var in variable_list:
    ds = xr.open_dataset(os.path.join(merra_eof_path, '%s.nc' % var))
    num_eofs[var] = ds.dims['n_components']
    ds.close()

fig = plt.figure(figsize=(10,10))
ax = plt.axes()
ax.barh(np.arange(len(variable_list)), [x for x in num_eofs.values()])
ax.set_yticks(np.arange(len(variable_list)))
ax.set_yticklabels(variable_list)
ax.set_title('Number of EOFs needed to explain 90% variance')
fig.savefig('eofs.png')
plt.close(fig)

