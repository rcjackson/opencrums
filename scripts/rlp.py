import tensorflow as tf
import pandas as pd
import xarray as xr
import numpy as np
from glob import glob
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K
import cartopy.crs as ccrs
import pickle
import sys

class LayerwiseRelevancePropogation():
    def __init__(self, model):
        self.model = model
        # Epsilon for epsilon RLP rule
        self.epsilon = 1e-5
        # Count the number of input layers --> Number of CNN networks
        self.no_inputs = 0
        self.pooling_type = "max"
        for x in model.layers:
            if 'input_' in x.name:
                self.no_inputs += 1
        self.alpha = 3
        self.beta = 1
        
    def backprop_dense(self, x, w, r):
        w_pos = tf.maximum(w, tf.zeros_like(w))
        z = tf.matmul(x, w_pos) + self.epsilon
        s = r / z
        c_p = tf.matmul(s, tf.transpose(w_pos))

        w_neg = tf.minimum(w, tf.zeros_like(w))
        z = tf.matmul(x, w_neg) + self.epsilon
        s = r / z
        c_n = tf.matmul(s, tf.transpose(w_neg))
        return (self.alpha * c_p + self.beta * c_n) * x

    def backprop_flatten(self, x, r):
        return tf.reshape(r, tf.shape(x))
    
    def relprop_add(self, x, r):
        z = tf.add_n(x) + self.epsilon
        s = r / z
        c = s
        return c * x
        
    def relprop_batchnorm(self, x, r, w):
        gamma = w[0]
        beta = w[1]
        mmean = w[2]
        mvar = w[3]
        z = tf.nn.batch_normalization(x, mmean, mvar, beta, gamma, self.epsilon) + self.epsilon
        s = r / z
        with tf.GradientTape() as g:
            g.watch(s)
            R = tf.nn.batch_normalization(s, mmean, mvar, beta, gamma, self.epsilon)
        c = g.gradient(R, s)
        return c * x

    # Conv2D backpropogation 
    def relprop_conv(self, x, r, w, strides=(1, 1, 1, 1), padding='SAME'):
        if len(strides) == 2:
            strides = (1, strides[0], strides[1], 1)
        w_pos = tf.maximum(w, 0.0)
        z = tf.nn.conv2d(x, w_pos, strides, padding) + self.epsilon
        s = r / z
        c_p = tf.compat.v1.nn.conv2d_backprop_input(tf.shape(x), w_pos, s, strides, padding)

        w_neg = tf.minimum(w, 0.0)
        z = tf.nn.conv2d(x, w_neg, strides, padding) + self.epsilon
        s = r / z
        c_n = tf.compat.v1.nn.conv2d_backprop_input(tf.shape(x), w_neg, s, strides, padding)

        return (self.alpha * c_p + self.beta * c_n) * x

    # Maxpooling layer RLP
    def relprop_pool(self, x, r, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME'):
        if len(strides) == 2:
            strides = (1, strides[0], strides[1], 1)
        if len(ksize) == 2:
            ksize = (1, ksize[0], ksize[1], 1)
        if self.pooling_type == 'avg':
            z = tf.nn.avg_pool(x, ksize, strides, padding) + self.epsilon
            s = r / z
            with tf.GradientTape() as g:
                g.watch(s)
                R = tf.nn.avg_pool(s, ksize, strides, padding)
            c = g.gradient(R, s)
        elif self.pooling_type == 'max':
            z = tf.nn.max_pool(x, ksize, strides, padding) + self.epsilon
            s = r / z
            c = tf.raw_ops.MaxPoolGradV2(orig_input=x, orig_output=z, grad=s, ksize=ksize, strides=strides, padding=padding)
        else:
            raise Exception('Error: no such unpooling operation.')
        return c * x

    def calc_relevance(self, inputs, outputs):
        if len(list(inputs.keys())) != self.no_inputs:
            raise InputError("Not enough inputs for model. Need %d inputs" % self.no_inputs)
        cur_layer = self.model.layers[-1]
        cur_relevance = outputs
        output_relevances = {}
        
        output_relevances["output"] = tf.constant(outputs.astype(np.float32))
        # Evaluate x for each layer:
        x_layers = inputs.copy()
        for my_layer in self.model.layers:
            if not isinstance(my_layer.input, list):
                x_layers[my_layer.name.split("/")[0]] = my_layer(x_layers[my_layer.input.name.split("/")[0]])
            else:
                x_layers[my_layer.name.split("/")[0]] = my_layer(list(x_layers[y.name.split("/")[0]] for y in my_layer.input))
                
        x_layers["output"] = tf.constant(outputs.astype(np.float32))
        # Backpropgate relevances
        for my_layer in self.model.layers[-1:0:-1]:
            if not isinstance(my_layer.input, list):
                inp_layer_name = my_layer.input.name.split("/")[0]
            
            if 'add' in my_layer.name:
                x_list = [x_layers[g.name.split("/")[0]] for g in my_layer.input]
                out_layer_name = my_layer._outbound_nodes[0].outputs.name.split("/")[0]
                add_relevances = tf.split(self.relprop_add(
                    x_list, output_relevances[out_layer_name]), self.no_inputs, axis=0)
                continue
            

            inp_layer = self.model.get_layer(inp_layer_name)
            try:
                out_layer_name = my_layer._outbound_nodes[0].outputs.name.split("/")[0]
            except IndexError:
                out_layer_name = "output"
            
            if 'dense' in my_layer.name or 'class' in my_layer.name:
                output_relevances[my_layer.name.split("/")[0]] = self.backprop_dense(
                    x_layers[inp_layer_name],
                    my_layer.trainable_weights[0],
                    output_relevances[out_layer_name])
            elif 'conv2d' in my_layer.name:
                
                output_relevances[my_layer.name.split("/")[0]] = self.relprop_conv(
                    x_layers[inp_layer_name],
                    output_relevances[out_layer_name],
                    my_layer.trainable_weights[0], 
                    strides=my_layer.strides, 
                    padding=my_layer.padding.upper())
            elif 'batch_normalization' in my_layer.name:
                output_relevances[my_layer.name.split("/")[0]] = self.relprop_batchnorm(
                        x_layers[inp_layer_name], 
                        output_relevances[out_layer_name],
                        my_layer.trainable_weights[0])
            elif 'flatten' in my_layer.name:
                name = my_layer.name.split("/")[0]
                if name[-1] == "n":
                    count = 0
                else:
                    count = int(name.split("_")[1])
                output_relevances[my_layer.name.split("/")[0]] = self.backprop_flatten(
                    x_layers[inp_layer_name],
                    add_relevances[count])
            elif 'max_pooling2d' in my_layer.name:
                output_relevances[my_layer.name.split("/")[0]] = self.relprop_pool(
                    x_layers[inp_layer_name],
                    output_relevances[out_layer_name], my_layer.pool_size,
                    my_layer.strides, my_layer.padding.upper())
            elif 'input' in my_layer.name:
                output_relevances[my_layer.name] = output_relevances[out_layer_name]
            
        return output_relevances

hour = 0
my_model = tf.keras.models.load_model(
        '/lcrc/group/earthscience/rjackson/opencrums/models/classifier-large-hou-lag-new-aqi-%d' % hour)

air_now_data = glob('/lcrc/group/earthscience/rjackson/epa_air_now/*.csv')
air_now_df = pd.concat(map(pd.read_csv, air_now_data))
air_now_df['datetime'] = pd.to_datetime(air_now_df['DateObserved'] + ' 00:00:00')
air_now_df = air_now_df.set_index('datetime')
air_now_df = air_now_df[air_now_df['ParameterName'] == "PM2.5"]
air_now_df = air_now_df.sort_index()
print(air_now_df['CategoryNumber'].values.min())

batch_size = 16

def get_air_now_label(time):
    if np.min(np.abs((air_now_df.index - time))) > timedelta(days=1):
        return np.nan
    ind = np.argmin(np.abs(air_now_df.index - time))
    if air_now_df['AQI'].values[ind] < 34.0:
        return 1
    elif air_now_df['AQI'].values[ind] >= 34.0 and air_now_df['AQI'].values[ind] < 42.0:
        return 2
    elif air_now_df['AQI'].values[ind] >= 42.0 and air_now_df['AQI'].values[ind] < 51.0:
        return 3
    elif air_now_df['AQI'].values[ind] >= 51.0 and air_now_df['AQI'].values[ind] < 57.0:
        return 4
    elif air_now_df['AQI'].values[ind] >= 57.0:
        return 5

# Load MERRA2 data from files
def load_data(species):
    ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_extended/%sCMASS*.nc' % species).sortby('time')
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
    ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_extended/%sFLUXU*.nc' % inp).sortby('time')
    x2 = ds["%sFLUXU" % inp].values
    scaler = StandardScaler()
    scaler.fit(np.reshape(x2, (old_shape[0], old_shape[1] * old_shape[2])))
    x2 = scaler.transform(
            np.reshape(x2, (old_shape[0], old_shape[1] * old_shape[2])))
    x2 = np.reshape(x2, old_shape)
    inputs[:, :, :, 1] = x2
    ds.close()
    ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_extended/%sFLUXV*.nc' % inp).sortby('time')
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
    times = times[where_valid]
    print(times.shape, classification.shape)
    y = np.concatenate([tf.one_hot(classification, 5).numpy(), times[:, np.newaxis]], axis=1)
    shape = inputs.shape
    x_dataset_train = {'input_%sMASS' % species: inputs[:, :, :, 0],
            'input_%sFLUXU' % inp: inputs[:, :, :, 1],
            'input_%sFLUXV' % inp: inputs[:, :, :, 2]}

    return x_dataset_train, y[:,:5], shape, y[:,5]

x_ds_train = {}
x_ds_test = {}
y_train = []
y_test = []
species_list = ['SS', 'SO4', 'SO2', 'OC','DU', 'DMS', 'BC']
for species in species_list:
    x_ds_train1, y_train, shape, t_train = load_data(species)
    x_ds_train.update(x_ds_train1)

for key in x_ds_train.keys():
    x_ds_train[key] = x_ds_train[key][:, :, :, np.newaxis]

y_predict = my_model.predict(x_ds_train)
y_predicts = y_predict.argmax(axis=1)

def slice_input_ds(x: dict, y, t, start, end):
    x_new = {}
    for key in x.keys():
        x_new[key] = x[key][start:end, :, :, :].copy()
    y_new = y[start:end, :].copy()
    t_test = t[start:end].copy()
    return x_new, y_new, t_test
print(x_ds_train['input_DMSMASS'].shape)
lrp = LayerwiseRelevancePropogation(my_model)
for i in range(0, y_predict.shape[0], 10):
    print(i + 10)
    x_slice, y_slice, t_slice = slice_input_ds(x_ds_train, y_predict, t_train, i, i + 10)
    relevance = lrp.calc_relevance(x_slice, y_slice)
    relevance['time'] = t_slice
    relevance['aqi'] = y_train
    with open('relevance_pickles_large_new/rel-%dhr%d.pickle' % (hour, i), mode='wb') as f:
        rkeys = list(relevance.keys())
        for k in rkeys:
            if "conv2d" in k or "flatten" in k or "batch_normalization" in k:
                del relevance[k]
            if "max_pooling2d" in k:
                del relevance[k]
        pickle.dump(relevance, f)

