import tensorflow as tf
import xarray as xr
import glob as glob
import numpy as np
import sys
import pickle
import xbatcher
#from hyperopt import hp

import ray
import ConfigSpace as cs

from datetime import timedelta, datetime
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Add, ReLU, Conv2DTranspose, Dense, Dropout, BatchNormalization, Lambda, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xbatcher.loaders.keras import CustomTFDataset
from deephyper.problem import HpProblem
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import LoggerCallback, SearchEarlyStopping
from deephyper.search.hps import CBO

species_list = ["DUCMASS", "DUSMASS", "DMSCMASS", "DMSSMASS", "DUFLUXU",
                "DUFLUXV", "SO4CMASS", "SO2CMASS", "SO4SMASS", "SO2SMASS",
                "OCCMASS", "OCSMASS", "OCFLUXU", "OCFLUXV", "SSFLUXU",
                "SSFLUXV", "SSCMASS", "SSSMASS", "BCCMASS", "BCSMASS",
                "BCFLUXU", "BCFLUXV"]
def preprocess_dataset(ds, species):
    x = ds[species].values
    old_shape = x.shape
    scaler = StandardScaler()
    scaler.fit(np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
    x = scaler.transform(
        np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
    x = np.reshape(x, old_shape)
    x = x[:, :, :, np.newaxis]
    x = tf.image.resize(x, (16, 32)).numpy()
    ds[species] = xr.DataArray(x, dims=('time', 'x', 'y', 'channel'))
    return ds

def load_data(species):
    preproc = lambda x: preprocess_dataset(x, species)
    ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/%s*.nc' % species,
                           preprocess=preproc).sortby('time')
    x_train, x_test = train_test_split(
            ds[species], test_size=0.20, random_state=3)
    return x_train, x_test


def classifier_model(shape, the_dict, dataset, species):
    width = shape[2]
    height = shape[1]
    input1 = Input(shape=(height, width, 1), name=species)
    
    mpool_1 = input1
    for i in range(the_dict['num_layers']):
        conv2d_1 = Conv2D(the_dict['num_channels'],
            (2, 2), activation=the_dict['activation'], padding='same',
            kernel_initializer='he_normal')(mpool_1)
        conv2d_1 = BatchNormalization()(conv2d_1)
        if i == the_dict['num_layers'] :
            name = "encoding"
        else:
            name = "layer_%d" % i
        mpool_1 = MaxPooling2D((2, 2), name=name)(conv2d_1)
 
    mpool_1 = Conv2D(1, (2, 2), activation=the_dict['activation'], padding='same',
            kernel_initializer='he_normal', name='encoding')(mpool_1)
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
    species = "BCCMASS"
    x_ds_train, x_ds_test = load_data(species)
    config['num_epochs'] = 10000
    config['activation'] = "relu"
    x_ds_train_gen = x_ds_train.batch.generator(
        input_dims={'x': 16, 'y': 32, 'channel': 1},
        batch_dims={'time': config["batch_size"]})
    x_ds_test_gen = x_ds_test.batch.generator(
        input_dims={'x': 16, 'y': 32, 'channel': 1},
        batch_dims={'time': config["batch_size"]})
    shape = x_ds_train.shape
    species = sys.argv[1]
    model = classifier_model(shape, config, x_ds_test, species)
    model.compile(optimizer=Adam(lr=config["learning_rate"]),
                  loss="mean_squared_error")
    model.summary()
    print(x_ds_train_gen[1])
    dataset_train = CustomTFDataset(x_ds_train_gen, x_ds_train_gen)
    dataset_test = CustomTFDataset(x_ds_test_gen, x_ds_test_gen)
    history = model.fit(
        dataset_train, validation_data=dataset_test,
        epochs=config["num_epochs"], callbacks=[EarlyStopping(patience=100,
        monitor="val_loss")])
    #model.save('../models/autoencoder-%s-lag-%d' % (variable, lag))
    return history.history['val_loss'][-1]


default_config = {
        "num_channels": 512,
        "learning_rate": 0.01,
        "num_layers": 4,
        "activation": "relu",
        "batch_size": 16}

#run(default_config)

problem = HpProblem()
problem.add_hyperparameter((1, 512, "log-uniform"), "num_channels")
problem.add_hyperparameter((0.0001, 0.1, "log-uniform"), "learning_rate")
problem.add_hyperparameter([1, 2, 3, 4], "num_layers")
problem.add_hyperparameter([8, 16, 32], "batch_size")

evaluator = Evaluator.create(run, method="ray", method_kwargs={
                     "address": None,
                     "num_gpus": 4,
                     "num_gpus_per_task": 1,
                     "callbacks": [SearchEarlyStopping(patience=10),
                         LoggerCallback()]})

search = CBO(problem, evaluator)
results = search.search(100)
