import tensorflow as tf
import sys
import ray
import xarray as xr
import numpy as np

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Add, ReLU, Conv2DTranspose, Dense
from tensorflow.distribute import MirroredStrategy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from sklearn.preprocessing import StandardScaler
from glob import glob
from deephyper.problem import HpProblem
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import LoggerCallback
from deephyper.search.hps import AMBS

variable_list = ["BCCMASS", "BCFLUXU", "BCFLUXV",
            "BCSMASS", "DMSCMASS", "DMSSMASS", 
            "DUCMASS", "DUCMASS25", "DUFLUXU", "DUFLUXV",
            "DUSMASS", "DUSMASS25", "OCCMASS", "OCFLUXU",
            "OCFLUXV", "OCSMASS", "SO2CMASS", "SO2SMASS",
            "SO4CMASS", "SO4SMASS", "SSCMASS", "SSCMASS25",
            "SSFLUXU", "SSFLUXV", "SSSMASS", "SSSMASS25",
            "SUFLUXU", "SUFLUXV"]


def multiencoder_model(shape, var, the_dict):
    width = shape[2]
    height = shape[1]
    inp_layer = Input(shape=(height, width, 3), name=var)
    for i in range(the_dict['num_layers']):
        conv2d_1 = Conv2D(the_dict['num_channels'],
                (the_dict['window_size'], the_dict['window_size']), activation=the_dict['activation'], padding='same',
                kernel_initializer='he_normal')(inp_layer)
        mpool_1 = MaxPooling2D((the_dict['window_size'], the_dict['window_size']))(conv2d_1)
    
    conv2d_1 = Conv2D(1, (the_dict['window_size'], the_dict['window_size']), activation=the_dict['activation'],
            padding='same',
            kernel_initializer='he_normal')(mpool_1)
    flat_1 = Flatten()(mpool_1)
    encoding = Dense(the_dict['num_dimensions'], name="encoding")(flat_1)
    encoding = Dense(height/2 * width/2)(encoding)
    dense_1 = Reshape(target_shape=(int(height/2), int(width/2), 1))(encoding)
    for i in range(the_dict['num_layers']):
        conv2d_1 = Conv2D(the_dict['num_channels'],
                (the_dict['window_size'], the_dict['window_size']), activation=the_dict['activation'], padding='same',
                kernel_initializer='he_normal')(dense_1)
        mpool_1 = UpSampling2D((the_dict['window_size'], the_dict['window_size']))(conv2d_1)
    output = Conv2D(3, (the_dict['window_size'], the_dict['window_size']), 
        activation=the_dict['activation'], padding='same', kernel_initializer='he_normal')(mpool_1)
    
    return Model(inp_layer, output)

tfrecords_path = '/lcrc/group/earthscience/rjackson/MERRA2/tfrecords/*.tfrecord'
normalizers_path = '/lcrc/group/earthscience/rjackson/opencrums/models/normalizers/'

def load_data():
    ds = xr.open_mfdataset(
            '/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/DUCMASS*.nc')
    x = ds["DUCMASS"].values
    old_shape = x.shape
    scaler = StandardScaler()
    scaler.fit(np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
    x = scaler.transform(np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
    x = np.reshape(x, old_shape)
    ds.close()
    ds = xr.open_mfdataset(
            '/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/DUFLUXU*.nc')
    y = ds["DUFLUXU"].values
    time = ds["time"].values
    print(time.shape)
    old_shape = x.shape
    scaler = StandardScaler()
    scaler.fit(np.reshape(y, (old_shape[0], old_shape[1] * old_shape[2])))
    y = scaler.transform(
            np.reshape(y, (old_shape[0], old_shape[1] * old_shape[2])))
    y = np.reshape(y, old_shape)
    ds.close()
    ds = xr.open_mfdataset(
            '/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/DUFLUXV*.nc')
    z = ds["DUFLUXV"].values
    old_shape = x.shape
    scaler = StandardScaler()
    scaler.fit(np.reshape(z, (old_shape[0], old_shape[1] * old_shape[2])))
    z = scaler.transform(
            np.reshape(z, (old_shape[0], old_shape[1] * old_shape[2])))
    z = np.reshape(z, old_shape)
    ds.close()
    x = np.stack([x, y, z], axis=-1)
    shape = x.shape
    return x, shape, time

def run(config: dict):
    x_ds, shape, time = load_data()
    model = multiencoder_model(shape, sys.argv[1], config)
    model.compile(optimizer=Adam(lr=config["learning_rate"]),
        loss="mean_squared_error", metrics=['mse'])
    history = model.fit(x_ds, x_ds, epochs=config["num_epochs"],
            batch_size=config["batch_size"])
    encoder = Model(model.input, model.get_layer("encoding").output)
    encodings = encoder.predict(x_ds)
    out_ds = xr.Dataset({'time': (['time'], time), 
        'encoding': (['time', 'latent'], encodings)})
    out_ds.to_netcdf('dust_encodings.nc')
    return history.history["mse"][-1]

default_config = {
        "num_epochs": 500,
        "num_channels": 64,
        "learning_rate": 1e-3,
        "num_dimensions": 8,
        "activation": "relu",
        "batch_size": 16,
        "num_layers": 2,
        "window_size": 2}

run(default_config)
