import tensorflow as tf
import pandas as pd
import xarray as xr
import glob as glob
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
#import ray

from datetime import timedelta, datetime
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Add, ReLU, Conv2DTranspose, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import RandomOverSampler

CBSA = "Houston, TX"

air_now_data = glob.glob('/lcrc/group/earthscience/rjackson/epa_air_now/*.csv')
air_now_df = pd.concat(map(pd.read_csv, air_now_data))
air_now_df['datetime'] = pd.to_datetime(air_now_df['DateObserved'] + ' 00:00:00')
air_now_df = air_now_df.set_index('datetime')
air_now_df = air_now_df.sort_index()
air_now_df = air_now_df[air_now_df['ParameterName'] == "PM2.5"]
print(air_now_df['CategoryNumber'].values.min())
lat_slice = slice(25, 35)
lon_slice = slice(-100, -90)


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
    ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/%s_daily/%sCMASS*.nc' % (site, species)).sel(lat=lat_slice, lon=lon_slice)
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
    classification = np.array(list(map(get_air_now_label, times)))
    where_valid = np.isfinite(classification)
    x = x[where_valid, :, :]
    classification = classification[where_valid] - 1
    y = tf.one_hot(classification, 2).numpy()
    x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.20, random_state=3)
    x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, test_size=0.50, random_state=3)
    shape = x.shape
    x_dataset_train = {'input_%sSMASS' % species: np.squeeze(x_train[:, :, :])}
    x_dataset_test = {'input_%sSMASS' % species: np.squeeze(x_test[:, :, :])}
    x_validation = {'input_%sSMASS' % species: np.squeeze(x_validation[:, :, :])}
     
    return x_dataset_train, x_dataset_test, x_validation, y_train, y_test, y_validation, shape



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
    
    model = load_model('../models/classifier-hou-lag-2class-surface-mass-only-daily-0') 

    y_predict_valid = model.predict(x_ds_valid)
    valid_confusion = confusion_matrix(y_valid.argmax(axis=1), 
            y_predict_valid.argmax(axis=1))
    #fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    cm_display = ConfusionMatrixDisplay(valid_confusion).plot()
    plt.gca().set_xticklabels(['Good', 'Moderate'])
    plt.gca().set_yticklabels(['Good', 'Moderate'])
    plt.savefig('validation_confusion_matrix_daily.png', bbox_inches='tight')
    plt.close()
    y_predict_train = model.predict(x_ds_train)
    valid_confusion = confusion_matrix(y_train.argmax(axis=1),
            y_predict_train.argmax(axis=1))
    #fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    cm_display = ConfusionMatrixDisplay(valid_confusion).plot()
    plt.gca().set_xticklabels(['Good', 'Moderate'])
    plt.gca().set_yticklabels(['Good', 'Moderate'])
    plt.savefig('training_confusion_matrix_daily.png', bbox_inches='tight')
    plt.close()
   
    y_predict_train = model.predict(x_ds_test)
    valid_confusion = confusion_matrix(y_test.argmax(axis=1),
            y_predict_train.argmax(axis=1))
    #fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    cm_display = ConfusionMatrixDisplay(valid_confusion).plot()
    plt.gca().set_xticklabels(['Good', 'Moderate'])
    plt.gca().set_yticklabels(['Good', 'Moderate'])
    plt.savefig('testing_confusion_matri_dailyx.png', bbox_inches='tight')
    plt.close()

    
default_config = {
        "num_epochs": 5000,
        "num_channels": 32,
        "learning_rate": 0.0001,
        "num_dense_nodes": 64,
        "num_dense_layers": 3,
        "activation": "relu",
        "batch_size": 16,
        "num_layers": 4}


if __name__ == "__main__":
    run(default_config)


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
