import tensorflow as tf
import sys
import ray
import xarray as xr
import numpy as np

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Add, ReLU, Conv2DTranspose, Dense, Dropout, MaxPooling3D, UpSampling3D
from tensorflow.keras.layers import TimeDistributed, LSTM, RepeatVector, ConvLSTM2D, Concatenate
from tensorflow.keras.applications import VGG19
from tensorflow.distribute import MirroredStrategy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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


def conv_block(input, num_filters, upsampling=False):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    if not upsampling:
        x = MaxPooling2D((2, 2))(x)
    else:
        x = UpSampling2D((2, 2))(x)
    return x


def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    if skip_features is not None:
        x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters, True)
    return x


def build_vgg19_unet(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained VGG19 Model """
    s1 = conv_block(inputs, 64)
    s2 = conv_block(s1, 128)
    s3 = conv_block(s2, 256)
    s4 = conv_block(s3, 512)

    """ Encoder """
    b1 = s4
    
    """ Decoder """
    d1 = decoder_block(b1, None, 512)
    d2 = decoder_block(d1, s4, 256)
    d3 = decoder_block(d2, s3, 128)
    d4 = decoder_block(d3, s2, 64)

    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation="relu")(d4)

    model = Model(inputs, outputs, name="VGG19_U-Net")
    return model

#def neighborhood_mse(y_true, y_pred):
#    strides = 1
#    num_points = 4
#    filters = np.ones((num_points, num_points,
#        y_true.shape[-1], y_true.shape[-1]))
#    conv = tf.nn.conv2d((y_true - y_pred) ** 2, filters,
#            strides, padding='SAME') / num_points ** 2
#    return tf.reduce_sum(conv)

code = 'HOU'
if code == 'HOU':
    ax_extent = [-105, -85, 25, 35]
elif code == 'SEUS':
    ax_extent = [-90, -75, 30, 37.5]

def load_data(num_timesteps):
    ds = xr.open_mfdataset(
            '/lcrc/group/earthscience/rjackson/era5-preprocessed/*.nc')
    ds.load()
    ds = ds.sortby('time')
    print(ds)
    x = np.squeeze(ds["u"].values)
    old_shape = x.shape
    # Make sure that we can divide our dataset into the given intervals
    #x = x[:(int(old_shape[0] / num_timesteps) * num_timesteps), :, :]
    #old_shape = x.shape
    #new_shape = (int(old_shape[0]/num_timesteps), num_timesteps,
    #    old_shape[1], old_shape[2])
    
    y = np.squeeze(ds["v"].values)
    which_lats = np.where(np.logical_and(ds["latitude"].values >= ax_extent[2], 
        ds["latitude"].values <= ax_extent[3]))[0]
    which_lons = np.where(np.logical_and(ds["longitude"].values >= ax_extent[0],
        ds["longitude"].values <= ax_extent[1]))[0]
    x = x[:,which_lats[0]:(which_lats[0]+96),which_lons[0]:(which_lons[0]+96)]
    y = y[:,which_lats[0]:(which_lats[0]+96),which_lons[0]:(which_lons[0]+96)]

    old_shape = y.shape
    #y = y[:(int(old_shape[0] / num_timesteps) * num_timesteps), :, :]
    x = np.stack([x, y], axis=-1)
    x_train, x_test = train_test_split(x, test_size=0.20)
    #    x_train = tf.keras.applications.vgg19.preprocess_input(x_train)
    #x_test = tf.keras.applications.vgg19.preprocess_input(x_test)
    x_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train))
    x_test = tf.data.Dataset.from_tensor_slices((x_test, x_test))
    
    shape = x.shape
    ds.close()
    del ds, x, y
    return x_dataset, shape, x_test


def run(config: dict):
    x_ds, shape, x_test = load_data(1)
    model = build_vgg19_unet(shape[1:])
    model.summary()
    x_ds = x_ds.batch(config["batch_size"])
    x_test = x_test.batch(config["batch_size"])
    model.compile(optimizer=Adam(lr=config["learning_rate"]),
        loss='mean_squared_error', metrics=['mse'])
    history = model.fit(x_ds, epochs=config["num_epochs"], 
            validation_data=x_test, callbacks=[EarlyStopping(patience=50)])
    model.save('../models/default_lstm_%s' % sys.argv[1]) 
    return history.history["mse"][-1]

default_config = {
        "num_epochs": 1000,
        "num_channels": 232,
        "learning_rate": 1e-2,
        "batch_size": 16}

objective_default = run(default_config)
print(f"MSE Default Configuration:  {objective_default:.3f}")


