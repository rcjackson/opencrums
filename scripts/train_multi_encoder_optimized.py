import tensorflow as tf
import sys
import xarray as xr
import numpy as np
import sys

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Add, ReLU, Conv2DTranspose, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
    conv2d_1 = Conv2D(int(the_dict['num_channels']/4), (2, 2), activation=the_dict['activation'],
            padding='same',
            kernel_initializer='he_normal', name="encoding")(mpool_1)
    conv2d_1 = Conv2D(the_dict['num_channels'],
            (2, 2), activation=the_dict['activation'], padding='same',
            kernel_initializer='he_normal')(conv2d_1)
    mpool_1 = UpSampling2D((2, 2))(conv2d_1)
    conv2d_1 = Conv2D(the_dict['num_channels'],
            (2, 2), activation=the_dict['activation'], padding='same',
            kernel_initializer='he_normal')(mpool_1)
    mpool_1 = UpSampling2D((2, 2))(conv2d_1)
    output = Conv2D(1, (2, 2), activation=the_dict['activation'],
            padding='same',
            kernel_initializer='he_normal')(mpool_1)
    
    return Model(inp_layer, output)


var = sys.argv[1]
tfrecords_path = '/lcrc/group/earthscience/rjackson/MERRA2/tfrecords/*.tfrecord'
normalizers_path = '/lcrc/group/earthscience/rjackson/opencrums/models/normalizers/'


def load_data():
    ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/%s*.nc' % var)
    x = ds[var].values
    old_shape = x.shape
    scaler = StandardScaler()
    scaler.fit(np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
    x = scaler.transform(np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
    x = np.reshape(x, old_shape)
    x_train, x_test = train_test_split(x)
    x_train = x_train[:, :, :, np.newaxis]
    x_test = x_test[:, :, :, np.newaxis]
    print(x_test.shape)
    x_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train))
    x_valid = tf.data.Dataset.from_tensor_slices((x_test, x_test))
    shape = x_train.shape
    return x_dataset, x_valid, shape


def run(config: dict):
    x_ds, x_valid, shape = load_data()
    model = multiencoder_model(shape, sys.argv[1], config)
    x_ds = x_ds.batch(config["batch_size"])
    x_valid = x_valid.batch(config["batch_size"])
    model.compile(optimizer=Adam(lr=config["learning_rate"]),
        loss="mean_squared_error", metrics=['mse'])
    history = model.fit(x_ds, epochs=config["num_epochs"], validation_data=x_valid,
        callbacks=EarlyStopping(patience=200, restore_best_weights=True))
    model.save('../models/encoder-decoder-%s' % var)
    return history.history["mse"][-1]


default_config = {
        "num_epochs": 10000,
        "num_channels": 64,
        "learning_rate": 0.0001,
        "num_dimensions": 17,
        "activation": "relu",
        "batch_size": 32}

run(default_config)
