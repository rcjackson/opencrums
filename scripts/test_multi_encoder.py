import tensorflow as tf
import sys
import xarray as xr
import matplotlib.pyplot as plt
import os
import numpy as np

from tensorflow.distribute import MirroredStrategy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from glob import glob

variable_list = ["BCCMASS", "BCFLUXU", "BCFLUXV",
            "BCSMASS", "DMSCMASS", "DMSSMASS", 
            "DUCMASS", "DUCMASS25", "DUFLUXU", "DUFLUXV",
            "DUSMASS", "DUSMASS25", "OCCMASS", "OCFLUXU",
            "OCFLUXV", "OCSMASS", "SO2CMASS", "SO2SMASS",
            "SO4CMASS", "SO4SMASS", "SSCMASS", "SSCMASS25",
            "SSFLUXU", "SSFLUXV", "SSSMASS", "SSSMASS25",
            "SUFLUXU", "SUFLUXV"]

tfrecords_path = '/lcrc/group/earthscience/rjackson/MERRA2/tfrecords/*.tfrecord'
model_path = '/lcrc/group/earthscience/rjackson/opencrums/models/multiencoder/'
out_path = '/lcrc/group/earthscience/rjackson/opencrums/models/multiencoder_test_pngs/'

if not os.path.exists(out_path):
    os.makedirs(out_path)

ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/%s*.nc' % sys.argv[1])
print(ds)
x = ds[sys.argv[1]].values
old_shape = x.shape
scaler = StandardScaler()
scaler.fit(np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
x = scaler.transform(np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
x = np.reshape(x, old_shape)
model_path = os.path.join(model_path, 'encoder-%s-%03d-hou.hdf5' % (
    sys.argv[1], int(sys.argv[2])))

num_images = 1000
strategy = MirroredStrategy()
with strategy.scope():
    model = load_model(model_path)
    i = 0
    while i < num_images:
        my_images = x[i:i+16, :, :]
        predict_images = model.predict(my_images)
        for j in range(my_images.shape[0]):
            fig, ax = plt.subplots(2, 1, figsize=(10, 5))
            img_valid = np.squeeze(my_images[j, :, :])
            img_predict = np.squeeze(predict_images[j, :, :])
            c = ax[0].pcolormesh(img_valid)
            ax[0].set_title('%s original' % sys.argv[1])
            plt.colorbar(c, ax=ax[0])
            ax[1].pcolormesh(img_predict)
            ax[1].set_title('%s predicted' % sys.argv[1])
            plt.colorbar(c, ax=ax[1])
            fig.savefig(os.path.join(out_path, '%sencoding%d.png' % (sys.argv[1], i)))
            i = i + 1
            plt.close(fig)
            del fig, ax

