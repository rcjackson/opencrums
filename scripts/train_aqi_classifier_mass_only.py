import tensorflow as tf
import pandas as pd
import xarray as xr
import glob as glob
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt

from datetime import timedelta, datetime
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Add, ReLU, Conv2DTranspose, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import max_norm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

CBSA = "Houston, TX"

air_now_data = glob.glob('/lcrc/group/earthscience/rjackson/epa_air_now/*.csv')
air_now_df = pd.concat(map(pd.read_csv, air_now_data))
air_now_df['datetime'] = pd.to_datetime(air_now_df['DateObserved'] + ' 00:00:00')
air_now_df = air_now_df.set_index('datetime')
air_now_df = air_now_df.sort_index()
air_now_df = air_now_df[air_now_df['ParameterName'] == "PM2.5"]
print(air_now_df['CategoryNumber'].values.min())

lag = int(sys.argv[1])
lat_slice = slice(25, 35)
lon_slice = slice(-100, -90)

sample_indices = None
def get_air_now_label(time):
    if np.min(np.abs((air_now_df.index - time))) > timedelta(days=1):
        return np.nan
    ind = np.argmin(np.abs(air_now_df.index - time))
    cat_number = air_now_df['CategoryNumber'].values[ind]
    if cat_number > 2:
        cat_number = np.nan
    return cat_number
site = "hou"

def load_data(species):
    ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/%s_extended/%sSMASS*.nc' % (site, species)).sel(lat=lat_slice, lon=lon_slice)
    print(ds)
    times = np.array(list(map(pd.to_datetime, ds.time.values)))
    try:
        x = ds["%sSMASS25" % species].values * 1e9
    except KeyError:
        x = ds["%sSMASS" % species].values * 1e9

    old_shape = x.shape
    scaler = StandardScaler()
    scaler.fit(np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
    x = scaler.transform(
            np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
    x = np.reshape(x, old_shape)
    classification = np.array(list(map(get_air_now_label, times)))
    where_valid = np.isfinite(classification)
    x = x[where_valid, :, :]
    classification = classification[where_valid] - 1
    y = tf.one_hot(classification, 2).numpy()
    # Ensure that oversampling indicies are the same for each feature
    x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.20, random_state=3)
    #print(sample_indices)
    if sample_indices is None:
        ros = RandomOverSampler(random_state=42)
        shape = x_train.shape
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
        x_train, y_train = ros.fit_resample(x_train, y_train.argmax(axis=1))
        y_train = tf.one_hot(y_train, 2).numpy()
        
        globals()["sample_indices"] = ros.sample_indices_
        x_train = x_train.reshape((x_train.shape[0], shape[1], shape[2]))
    else:
        x_train = x_train[sample_indices, :, :]
        y_train = y_train[sample_indices, :]

    x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, test_size=0.50, random_state=3)
    x_dataset_train = {'input_%ssmass' % species.lower(): np.squeeze(x_train)}
    x_dataset_test = {'input_%ssmass' % species.lower(): np.squeeze(x_test)}
    x_validation = {'input_%ssmass' % species.lower(): np.squeeze(x_validation)}
    shape = x_train.shape
    return x_dataset_train, x_dataset_test, x_validation, y_train, y_test, y_validation, shape


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
            if i == 0:
                val = 10
            else:
                val = 0.01
            conv2d_1 = Conv2D(the_dict['num_channels'],
                (2, 2), activation=the_dict['activation'], padding='same',
                kernel_initializer='he_normal', kernel_constraint=max_norm(2),
                kernel_regularizer=l2(val))(mpool_1)
            conv2d_1 = BatchNormalization()(conv2d_1)
            mpool_1 = MaxPooling2D((2, 2))(conv2d_1)
        mpool_1s.append(Flatten()(mpool_1))
    flat_1 = Add()(mpool_1s)

    for i in range(the_dict['num_dense_layers']):
        flat_1 = Dense(the_dict['num_dense_nodes'], activation='relu',
                kernel_regularizer=l2(val), 
                kernel_constraint=max_norm(2)
                )(flat_1)
        flat_1 = BatchNormalization()(flat_1)

    output = Dense(2, activation="softmax", name="class")(flat_1)

    return Model(in_layers, output)



def run(config: dict):
    x_ds_train = {}
    x_ds_test = {}
    x_ds_valid = {}
    y_train = []
    y_test = []
    y_valid = []
    species_list = ['SS', 'SO4', 'OC', 'DU', 'BC']
    for species in species_list:
        print(species)
        x_ds_train1, x_ds_test1, x_ds_valid1, y_train, y_test, y_valid, shape = load_data(species)
        x_ds_train.update(x_ds_train1)
        x_ds_test.update(x_ds_test1)
        x_ds_valid.update(x_ds_valid1)
    model = classifier_model(shape, config, x_ds_test)
    model.compile(optimizer=Adam(learning_rate=config['learning_rate']),
        loss="binary_crossentropy", metrics=['acc'])
    model.summary()
    #class_weights = {0: 1 / no_good * (total / 2.0), 1: 1 / no_moderate * (total / 2.0)}
    # AQI classes inbalanced, need weights
    history = model.fit(
            x_ds_train, y_train, 
            validation_data=(x_ds_test, y_test), epochs=config["num_epochs"],
            batch_size=config["batch_size"],
            callbacks=[EarlyStopping(patience=50, monitor="val_acc"), ReduceLROnPlateau(min_lr=1e-8,
                factor=0.2, patience=5)])
    model.save('../models/classifier-mass-only-%d-nodes-%dconv-' % 
            (config["num_dense_nodes"], config["num_layers"]))
    fig = plt.figure()
    y_predict_valid = model.predict(x_ds_valid).argmax(axis=1)    
    y_valid = y_valid.argmax(axis=1)
    cm = confusion_matrix(y_valid, y_predict_valid)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=["Good", "Moderate"])
    disp.plot()
    plt.savefig('validation_confusion_matrix_%d-nodes-%dconv.png' %
        (config["num_dense_nodes"], config["num_layers"]))
    plt.close(fig)

    fig = plt.figure()
    y_test_pred = model.predict(x_ds_test).argmax(axis=1)
    y_test = y_test.argmax(axis=1)
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=["Good", "Moderate"])
    disp.plot()
    plt.savefig('testing_confusion_matrix_%d-nodes-%dconv.png' %
        (config["num_dense_nodes"], config["num_layers"]))
    plt.close(fig)

    fig = plt.figure()
    y_train_pred = model.predict(x_ds_train).argmax(axis=1)
    y_train = y_train.argmax(axis=1)
    cm = confusion_matrix(y_train, y_train_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=["Good", "Moderate"])
    disp.plot()
    plt.savefig('training_confusion_matrix_%d-nodes-%dconv.png' %
            (config["num_dense_nodes"], config["num_layers"]))
    plt.close(fig)

    return history.history


default_config = {
        "num_epochs": 1000,
        "num_channels": 4,
        "learning_rate": 3e-4,
        "num_dense_nodes": int(sys.argv[1]),
        "num_dense_layers": 3,
        "activation": "relu",
        "batch_size": 16,
        "num_layers": 2}



#if not ray.is_initialized():
#    ray.init(num_cpus=128, num_gpus=8, log_to_driver=False)
if __name__ == "__main__":
    run(default_config)

