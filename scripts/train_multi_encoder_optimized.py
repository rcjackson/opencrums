import tensorflow as tf
import sys
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
    inp_layer = Input(shape=(height, width, 1), name=var)
    conv2d_1 = Conv2D(the_dict['num_channels'],
            (2, 2), activation=the_dict['activation'], padding='same',
            kernel_initializer='he_normal')(inp_layer)
    mpool_1 = MaxPooling2D((2, 2))(conv2d_1)
    conv2d_1 = Conv2D(the_dict['num_channels'],
            (2, 2), activation=the_dict['activation'], padding='same',
            kernel_initializer='he_normal')(mpool_1)
    mpool_1 = MaxPooling2D((2, 2))(conv2d_1)
    conv2d_1 = Conv2D(1, (2, 2), activation=the_dict['activation'],
            padding='same',
            kernel_initializer='he_normal')(mpool_1)
    flat_1 = Flatten()(mpool_1)
    encoding = Dense(the_dict['num_dimensions'], name="encoding")(flat_1)
    encoding = Dense(height/4 * width/4)(encoding)
    dense_1 = Reshape(target_shape=(int(height/4), int(width/4), 1))(encoding)
    conv2d_1 = Conv2D(the_dict['num_channels'],
            (2, 2), activation=the_dict['activation'], padding='same',
            kernel_initializer='he_normal')(dense_1)
    mpool_1 = UpSampling2D((2, 2))(conv2d_1)
    conv2d_1 = Conv2D(the_dict['num_channels'],
            (2, 2), activation=the_dict['activation'], padding='same',
            kernel_initializer='he_normal')(mpool_1)
    mpool_1 = UpSampling2D((2, 2))(conv2d_1)
    output = Conv2D(1, (2, 2), activation=the_dict['activation'],
            padding='same',
            kernel_initializer='he_normal')(mpool_1)
    
    return Model(inp_layer, output)

tfrecords_path = '/lcrc/group/earthscience/rjackson/MERRA2/tfrecords/*.tfrecord'
normalizers_path = '/lcrc/group/earthscience/rjackson/opencrums/models/normalizers/'
def load_data():
    ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/DUFLUXU*.nc')
    print(ds)
    x = ds["DUFLUXU"].values
    old_shape = x.shape
    scaler = StandardScaler()
    scaler.fit(np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
    x = scaler.transform(np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
    x = np.reshape(x, old_shape)
    x_dataset = tf.data.Dataset.from_tensor_slices((x, x))
    shape = x.shape
    return x_dataset, shape


def run(config: dict):
    x_ds, shape = load_data()
    model = multiencoder_model(shape, sys.argv[1], config)
    x_ds = x_ds.batch(config["batch_size"])
    model.compile(optimizer=Adam(lr=config["learning_rate"]),
        loss="mean_squared_error", metrics=['mse'])
    history = model.fit(x_ds, epochs=config["num_epochs"])
    model.save('../models/encoder-decoder-DUFLUXU')
    return history.history["mse"][-1]

default_config = {
        "num_epochs": 120,
        "num_channels": 50,
        "learning_rate": 0.00314,
        "num_dimensions": 17,
        "activation": "selu",
        "batch_size": 33}

run(default_config)
