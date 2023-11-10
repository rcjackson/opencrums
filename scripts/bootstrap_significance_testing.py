import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import bootstrap

ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/DUMASS*.nc
