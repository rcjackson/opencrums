import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray as xr
import pandas as pd
import numpy as np
import sys

from datetime import timedelta
from sklearn.metrics import confusion_matrix
from train_aqi_classifier_all_species import load_data, get_air_now_label
from tensorflow.keras.models import load_model



def run():
    x_ds_train = {}
    x_ds_test = {}
    x_ds_valid = {}
    y_train = []
    y_test = []
    y_valid = []
    species_list = ['SS', 'SO4', 'SO2', 'OC', 'DU', 'DMS', 'BC']
    for species in species_list:
        print(species)
        x_ds_train1, x_ds_test1, x_ds_valid1, y_train, y_test, y_valid, shape = load_data(species)
        x_ds_train.update(x_ds_train1)
        x_ds_test.update(x_ds_test1)
        x_ds_valid.update(x_ds_valid1)

    model = load_model('../models/classifier-large-hou-lag-new-aqi-0')
    y_valid_predict = model.predict(x_ds_valid1).argmax(axis=1)
    y_valid = y_valid.argmax(axis=1) 
    matrix = confusion_matrix(y_valid_predict, y_valid)
    fig = plt.figure(figsize=(7,7))
    ax = plt.axes()
    ax.matshow(matrix, cmap='Reds')
    for (i, j), z in np.ndenumerate(matrix):
        ax.text(j, i, '{:d}'.format(int(z)), ha='center', va='center', color='k', fontsize=16)
    ax.set_xticklabels(['', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    ax.set_yticklabels(['', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    fig.savefig('confusion_matrix_aqi_%dhr.png' % hour)

if __name__ == "__main__":
    run()
