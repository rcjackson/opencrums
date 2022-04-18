import tensorflow as tf
import pandas as pd
import xarray as xr
import glob as glob
import numpy as np
import sys
import pickle

from datetime import timedelta, datetime
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Add, ReLU, Conv2DTranspose, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

air_now_data = glob.glob('/lcrc/group/earthscience/rjackson/epa_air_now/*.csv')
air_now_df = pd.concat(map(pd.read_csv, air_now_data))
air_now_df['datetime'] = pd.to_datetime(air_now_df['DateObserved'] + ' 00:00:00')
air_now_df = air_now_df.set_index('datetime')
air_now_df = air_now_df.sort_index()
print(air_now_df['CategoryNumber'].values.min())

def get_air_now_label(time):
    if np.min(np.abs((air_now_df.index - time))) > timedelta(days=1):
        return np.nan
    ind = np.argmin(np.abs(air_now_df.index - time))
    return air_now_df['CategoryNumber'].values[ind]


def load_data(species):
    ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/%sCMASS*.nc' % species).sortby('time')
    print(ds)
    times = np.array(list(map(pd.to_datetime, ds.time.values)))
    x = ds["%sCMASS" % species].values
    old_shape = x.shape
    scaler = StandardScaler()
    scaler.fit(np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
    x = scaler.transform(
            np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
    x = np.reshape(x, old_shape)
    inputs = np.zeros((old_shape[0], old_shape[1], old_shape[2], 3))
    inputs[:, :, :, 0] = x
    ds.close()
    if species == "SO4" or species == "DMS" or species == "SO2":
       inp = "SU"
    else:
       inp = species
    ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/%sFLUXU*.nc' % inp).sortby('time')
    x2 = ds["%sFLUXU" % inp].values
    scaler = StandardScaler()
    scaler.fit(np.reshape(x2, (old_shape[0], old_shape[1] * old_shape[2])))
    x2 = scaler.transform(
            np.reshape(x2, (old_shape[0], old_shape[1] * old_shape[2])))
    x2 = np.reshape(x2, old_shape)
    inputs[:, :, :, 1] = x2
    ds.close()
    ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/%sFLUXV*.nc' % inp).sortby('time')
    x2 = ds["%sFLUXV" % inp].values
    scaler = StandardScaler()
    scaler.fit(np.reshape(x2, (old_shape[0], old_shape[1] * old_shape[2])))
    x2 = scaler.transform(
            np.reshape(x2, (old_shape[0], old_shape[1] * old_shape[2])))
    x2 = np.reshape(x2, old_shape)
    inputs[:, :, :, 2] = x2
    classification = np.array(list(map(get_air_now_label, times)))
    where_valid = np.isfinite(classification)
    inputs = inputs[where_valid, :, :, :]
    classification = classification[where_valid]
    y = tf.one_hot(classification, 5).numpy()
    shape = inputs.shape
    x_dataset_train = {'input_%sMASS' % species: np.squeeze(inputs[:, :, :, 0]),
            'input_%sFLUXU' % inp: np.squeeze(inputs[:, :, :, 1]),
            'input_%sFLUXV' % inp: np.squeeze(inputs[:, :, :, 2])}

    return x_dataset_train, y, times


def classifier_model(shape, the_dict, dataset):
    width = shape[2]
    height = shape[1]
    mpool_1s = []
    in_layers = []
    dict_keys = list(dataset.keys())
    for j in range(len(dict_keys)):
        inp_layer = Input(shape=(height, width, 1), name=dict_keys[j])
        mpool_1 = inp_layer
        in_layers.append(inp_layer)
        for i in range(the_dict['num_layers']):
            conv2d_1 = Conv2D(the_dict['num_channels'],
                (2, 2), activation=the_dict['activation'], padding='same',
                kernel_initializer='he_normal')(mpool_1)
            conv2d_1 = BatchNormalization()(conv2d_1)
            mpool_1 = MaxPooling2D((2, 2))(conv2d_1)
        mpool_1s.append(Flatten()(mpool_1))
    flat_1 = Add()(mpool_1s)
    
    for i in range(the_dict['num_dense_layers']):
        flat_1 = Dense(the_dict['num_dense_nodes'], activation='relu'
                )(flat_1)
        flat_1 = BatchNormalization()(flat_1)

    output = Dense(5, activation="softmax", name="class")(flat_1)
    
    return Model(in_layers, output)


def run(config: dict):
    x_ds_train = {}
    x_ds_test = {}
    y_train = []
    y_test = []
    for species in config["species_list"]:
        print(species)
        x_ds_train1, y, times = load_data(species)
        x_ds_train.update(x_ds_train1)
    
    # AQI classes inbalanced, need weights
    model = load_model('../models/classifier')
    y_predict = model.predict(x_ds_train).argmax(axis=1)
    y = y.argmax(axis=1)
    ds = xr.Dataset({'time': times, 'y_predict': y_predict, 'y_true': y})        
    ds.to_netcdf('aqi_index.nc')

default_config = {
        "num_epochs": 200,
        "num_channels": 128,
        "learning_rate": 0.0001,
        "num_dense_nodes": 8,
        "num_dense_layers": 3,
        "activation": "relu",
        "batch_size": 10,
        "num_layers": 2,
        "species_list": ['SS', 'SO4', 'SO2', 'OC','DU', 'DMS', 'BC']}

history = run(default_config)
