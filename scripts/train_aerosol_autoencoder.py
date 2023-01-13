import tensorflow as tf
import pandas as pd
import xarray as xr
import glob as glob
import numpy as np
import sys
import pickle
#import ray

from datetime import timedelta, datetime
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Add, ReLU, Conv2DTranspose, Dense, Dropout, BatchNormalization, Lambda, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from deephyper.problem import HpProblem
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import LoggerCallback
from deephyper.search.hps import AMBS

CBSA = "Houston, TX"

air_now_data = glob.glob('/lcrc/group/earthscience/rjackson/epa_air_now/*.csv')
air_now_df = pd.concat(map(pd.read_csv, air_now_data))
air_now_df['datetime'] = pd.to_datetime(air_now_df['DateObserved'] + ' 00:00:00')
air_now_df = air_now_df.set_index('datetime')
air_now_df = air_now_df.sort_index()
air_now_df = air_now_df[air_now_df['ParameterName'] == "PM2.5"]
print(air_now_df['CategoryNumber'].values.min())

lag = int(sys.argv[1])

def get_air_now_label(time):
    if np.min(np.abs((air_now_df.index - time))) > timedelta(days=1):
        return np.nan
    ind = np.argmin(np.abs(air_now_df.index - time))
    return air_now_df['CategoryNumber'].values[ind]

site = "hou"

def load_data(species):
    ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/%s_reduced/%sCMASS*.nc' % (site, species)).sortby('time')
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
    ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/%s_reduced/%sFLUXU*.nc' % (site, inp)).sortby('time')
    x2 = ds["%sFLUXU" % inp].values
    scaler = StandardScaler()
    scaler.fit(np.reshape(x2, (old_shape[0], old_shape[1] * old_shape[2])))
    x2 = scaler.transform(
            np.reshape(x2, (old_shape[0], old_shape[1] * old_shape[2])))
    x2 = np.reshape(x2, old_shape)
    inputs[:, :, :, 1] = x2
    ds.close()
    ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/%s_reduced/%sFLUXV*.nc' % (site, inp)).sortby('time')
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
    classification = classification[where_valid] - 1
    if lag > 0:
        classification = classification[lag:]
        inputs = inputs[:(-lag), :, : :]
    print(inputs.shape, classification.shape)
    y = tf.one_hot(classification, 5).numpy()
    x_train, x_test, y_train, y_test = train_test_split(
            inputs, y, test_size=0.20, random_state=3)
    shape = inputs.shape
    x_dataset_train = {'input_%sMASS' % species: np.squeeze(x_train[:, :, :, 0]),
            'input_%sFLUXU' % inp: np.squeeze(x_train[:, :, :, 1]),
            'input_%sFLUXV' % inp: np.squeeze(x_train[:, :, :, 2])}
    x_dataset_test = {'input_%sMASS' % species: np.squeeze(x_test[:, :, :, 0]),
            'input_%sFLUXU' % inp: np.squeeze(x_test[:, :, :, 1]),
            'input_%sFLUXV' % inp: np.squeeze(x_test[:, :, :, 2])}

    return x_dataset_train, x_dataset_test, y_train, y_test, shape


def classifier_model(shape, the_dict, dataset):
    width = shape[2]
    height = shape[1]
    mpool_1s = []
    in_layers = []
    dict_keys = list(dataset.keys())
    num_variables = len([x for x in dataset.keys()])
    input1 = Input(shape=(height, width, 1))
    
    mpool_1 = input1
    for i in range(the_dict['num_layers']):
        conv2d_1 = Conv2D(the_dict['num_channels'],
            (2, 2), activation=the_dict['activation'], padding='same',
            kernel_initializer='he_normal')(mpool_1)
        conv2d_1 = BatchNormalization()(conv2d_1)
        if i == the_dict['num_layers'] - 1:
            name = "encoding"
        else:
            name = "layer_%d" % i
        mpool_1 = MaxPooling2D((2, 2), name=name)(conv2d_1)
    orig_shape = mpool_1.shape
    flat_1 = Flatten()(mpool_1)


    
    for i in range(the_dict['num_layers']):
        conv2d_1 = Conv2DTranspose(the_dict['num_channels'],
            (2, 2), activation=the_dict['activation'], padding='same',
            kernel_initializer='he_normal')(mpool_1)
        conv2d_1 = BatchNormalization()(conv2d_1)
        mpool_1 = UpSampling2D((2, 2))(conv2d_1)
    output = Conv2D(1, (2, 2), 
            padding='same', 
            activation="sigmoid", kernel_initializer='he_normal')(mpool_1)

    return Model(input1, output)


def run(config: dict):
    x_ds_train = {}
    x_ds_test = {}
    y_train = []
    y_test = []
    species_list = ['SS', 'SO4', 'SO2', 'OC', 'DU', 'DMS', 'BC']
    for species in species_list:
        print(species)
        x_ds_train1, x_ds_test1, y_train, y_test, shape = load_data(species)
        x_ds_train.update(x_ds_train1)
        x_ds_test.update(x_ds_test1)
    
    #x_train = np.stack([x for x in x_ds_train.values()], axis=3)
    #x_test = np.stack([x for x in x_ds_test.values()], axis=3)
    var_keys = [x for x in x_ds_train.keys()]
    variable = var_keys[int(sys.argv[2])]
    x_train = x_ds_train[variable][:, :, :, np.newaxis]
    x_test = x_ds_test[variable][:, :, :, np.newaxis]
    with tf.device("/GPU:0"):
        x_train = tf.image.resize(x_train, (16, 32)).numpy()
        x_test = tf.image.resize(x_test, (16, 32)).numpy()
        print(x_train.shape, x_test.shape)
        print(y_train.shape, y_test.shape)
        shape = (None, 16, 32, 17)
        model = classifier_model(shape, config, x_ds_test)
        model.compile(optimizer=Adam(lr=config["learning_rate"]),
                loss="mean_squared_error")
        model.summary()

    # AQI classes inbalanced, need weights
        class_weight = {0: 1, 1: 2, 2: 20, 3: 90, 4: 510}
        history = model.fit(
                x_train, x_train, validation_data=(x_test, x_test),
             epochs=config["num_epochs"], callbacks=[EarlyStopping(patience=100,
                 monitor="val_loss")],
                batch_size=config["batch_size"])
        model.save('../models/autoencoder-%s-lag-%d' % (variable, lag))
    return history.history

default_config = {
        "num_epochs": 10000,
        "num_channels": 64,
        "learning_rate": 0.001,
        "num_dense_nodes": 64,
        "num_dense_layers": 3,
        "activation": "relu",
        "batch_size": 16,
        "num_layers": 3}


#if not ray.is_initialized():
#    ray.init(num_cpus=128, num_gpus=8, log_to_driver=False)

run(default_config)
#run_default = ray.remote(num_cpus=16, num_gpus=1)(run)
#objective_default = ray.get(run_default.remote(default_config))


#problem = HpProblem()
#problem.add_hyperparameter((20, 200), "num_epochs")
#problem.add_hyperparameter((8, 512, "log-uniform"), "num_dense_nodes")
#problem.add_hyperparameter((1, 2), "num_layers")
#problem.add_hyperparameter((1, 8), "num_dense_layers")
#problem.add_hyperparameter((4, 256, "log-uniform"), "num_channels")
# Categorical hyperparameter (sampled with uniform prior)
#ACTIVATIONS = [
#    "elu", "gelu", "hard_sigmoid", "linear", "relu", "selu",
#    "sigmoid", "softplus", "softsign", "swish", "tanh",
#]
#problem.add_hyperparameter(ACTIVATIONS, "activation")
#problem.add_hyperparameter((1e-5, 1e-2, "log-uniform"), "learning_rate")
#problem.add_hyperparameter((8, 256, "log-uniform"), "batch_size")

#method_kwargs = {
#        "num_cpus": 128,
#        "num_cpus_per_task": 16,
#        "callbacks": [LoggerCallback()]
#    }

#method_kwargs["num_cpus"] = 128
#method_kwargs["num_gpus"] = 4
#method_kwargs["num_cpus_per_task"] = 16
#method_kwargs["num_gpus_per_task"] = 1

#evaluator = Evaluator.create(run, method="ray", method_kwargs=method_kwargs)
#search = AMBS(problem, evaluator)
#results = search.search(50)
#results.to_csv('hpsearch_results_classifieraqi.csv')
