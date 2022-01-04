import tensorflow as tf
import sys
import ray
import xarray as xr
import numpy as np

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Add, ReLU, Conv2DTranspose, Dense, Dropout, BatchNormalization
from tensorflow.distribute import MirroredStrategy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
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


def classifier_model(shape, var, the_dict):
    width = shape[2]
    height = shape[1]
    mpool_1s = []
    in_layers = []
    for j in range(shape[3]):
        inp_layer = Input(shape=(height, width, 1), name="input_%d" % (j + 1))
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

    output = Dense(4, activation="softmax", name="class")(flat_1)
    
    return Model(in_layers, output)

tfrecords_path = '/lcrc/group/earthscience/rjackson/MERRA2/tfrecords/*.tfrecord'
normalizers_path = '/lcrc/group/earthscience/rjackson/opencrums/models/normalizers/'

def load_data():
    ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/DUCMASS*.nc')
    print(ds)
    x = ds["DUCMASS"].values
    old_shape = x.shape
    scaler = StandardScaler()
    scaler.fit(np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
    x = scaler.transform(
            np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
    x = np.reshape(x, old_shape)
    inputs = np.zeros((old_shape[0], old_shape[1], old_shape[2], 3))
    inputs[:, :, :, 0] = x
    ds.close()
    ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/DUFLUXU*.nc')
    x2 = ds["DUFLUXU"].values
    scaler = StandardScaler()
    scaler.fit(np.reshape(x2, (old_shape[0], old_shape[1] * old_shape[2])))
    x2 = scaler.transform(
            np.reshape(x2, (old_shape[0], old_shape[1] * old_shape[2])))
    x2 = np.reshape(x2, old_shape)
    inputs[:, :, :, 1] = x2
    ds.close()
    ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/DUFLUXV*.nc')
    x2 = ds["DUFLUXV"].values
    scaler = StandardScaler()
    scaler.fit(np.reshape(x2, (old_shape[0], old_shape[1] * old_shape[2])))
    x2 = scaler.transform(
            np.reshape(x2, (old_shape[0], old_shape[1] * old_shape[2])))
    x2 = np.reshape(x2, old_shape)
    inputs[:, :, :, 2] = x2
    class_ds = xr.open_dataset('classification_dust.nc')
    y = tf.one_hot(class_ds.classification.values, 4).numpy()
    print(x.shape)
    x_train, x_test, y_train, y_test = train_test_split(
            inputs, y, test_size=0.20)
    shape = inputs.shape
    x_dataset_train = {'input_1': np.squeeze(x_train[:, :, :, 0]),
            'input_2': np.squeeze(x_train[:, :, :, 1]),
            'input_3': np.squeeze(x_train[:, :, :, 2])}
    x_dataset_test = {'input_1': np.squeeze(x_test[:, :, :, 0]),
            'input_2': np.squeeze(x_test[:, :, :, 1]),
            'input_3': np.squeeze(x_test[:, :, :, 2])}

    return x_dataset_train, x_dataset_test, y_train, y_test, shape


def run(config: dict):
    x_ds_train, x_ds_test, y_train, y_test, shape = load_data()
    model = classifier_model(shape, sys.argv[1], config)
    model.compile(optimizer=Adam(lr=config["learning_rate"]),
        loss="categorical_crossentropy", metrics=['acc'])
    model.summary() 
    history = model.fit(
            x_ds_train, y_train, 
            validation_data=(x_ds_test, y_test), epochs=config["num_epochs"],
            batch_size=config["batch_size"])
    model.save('../models/classifier')
    return history.history["val_acc"][-1]

default_config = {
        "num_epochs": 100,
        "num_channels": 256,
        "learning_rate": 0.0001,
        "num_dense_nodes": 128,
        "num_dense_layers": 2,
        "activation": "relu",
        "batch_size": 220,
        "num_layers": 2}
run(default_config)
