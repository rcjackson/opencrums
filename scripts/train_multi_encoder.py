import tensorflow as tf
import sys
import xarray as xr
import numpy as np

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Add, ReLU, Conv2DTranspose, Dense
from tensorflow.distribute import MirroredStrategy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Model
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


def multiencoder_model(ds, var):
    shape = ds.shape
    width = shape[2]
    height = shape[1]
    inp_layer = Input(shape=(height, width, 1), name=var)
    
    
    encoding = Dense(8, activation="relu")(x)
    
    
    output = Reshape(target_shape=(int(height), int(width), 1))(out)
    
    return Model(inp_layer, output)

tfrecords_path = '/lcrc/group/earthscience/rjackson/MERRA2/tfrecords/*.tfrecord'
normalizers_path = '/lcrc/group/earthscience/rjackson/opencrums/models/normalizers/'
ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/%s*.nc' % sys.argv[1])
print(ds)
x = ds[sys.argv[1]].values
old_shape = x.shape
scaler = StandardScaler()
scaler.fit(np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
x = scaler.transform(np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
x = np.reshape(x, old_shape)
strategy = MirroredStrategy()
with strategy.scope():
    model = multiencoder_model(x, sys.argv[1])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()
    checkpointer = ModelCheckpoint(
        filepath=('/lcrc/group/earthscience/rjackson/opencrums/models/multiencoder/encoder-%s-{epoch:03d}-hou.hdf5' % sys.argv[1]),
        verbose=1)
    model.fit(x, x, epochs=2000, callbacks=[checkpointer])
        
        #model.save(normalizers_path + var)
        
    

