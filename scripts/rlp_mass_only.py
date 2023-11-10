import tensorflow as tf
import pandas as pd
import xarray as xr
import numpy as np
import os
from glob import glob
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K
from scipy.interpolate import interp1d
from distributed import LocalCluster, Client

import cartopy.crs as ccrs
import pickle
import sys

class LayerwiseRelevancePropogation():
    def __init__(self, model):
        self.model = model
        # Epsilon for epsilon RLP rule
        self.epsilon = 1e-10
        # Count the number of input layers --> Number of CNN networks
        self.no_inputs = 0
        self.pooling_type = "max"
        for x in model.layers:
            if 'input_' in x.name:
                self.no_inputs += 1
        self.alpha = 1
        self.beta = 1 - self.alpha
        
    def backprop_dense(self, x, w, b, r):
        w_pos = tf.maximum(w, 0)
        b_pos = tf.maximum(b, 0)
        w_neg = tf.minimum(w, 0)
        b_neg = tf.minimum(b, 0)
        z = tf.matmul(x, w_pos) + b_pos
        s = r / z
        c_p = tf.matmul(s, tf.transpose(w_pos))

        z = tf.matmul(x, w_neg) + b_neg
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
        
    def relprop_batchnorm(self, x, r, w, b):
        w_pos = tf.maximum(w, 0)
        b_pos = tf.maximum(b, 0)
        w_neg = tf.minimum(w, 0)
        b_neg = tf.minimum(b, 0)
        gamma = w_pos[0]
        beta = w_pos[1]
        mmean = w_pos[2]
        mvar = w_pos[3]
        b_pos = tf.maximum(b, 0)
        z = tf.nn.batch_normalization(x, mmean, mvar, beta, gamma, self.epsilon) + self.epsilon + b_pos
        s = r / z
        with tf.GradientTape() as g:
            g.watch(s)
            R = tf.nn.batch_normalization(s, mmean, mvar, beta, gamma, self.epsilon) + self.epsilon + b_pos
        c_p = g.gradient(R, s)

        gamma = w_neg[0]
        beta = w_neg[1]
        mmean = w_neg[2]
        mvar = w_neg[3]
        b_neg = tf.minimum(b, 0)
        z = tf.nn.batch_normalization(x, mmean, mvar, beta, gamma, self.epsilon) + b_neg
        s = r / z
        with tf.GradientTape() as g:
            g.watch(s)
            R = tf.nn.batch_normalization(s, mmean, mvar, beta, gamma, self.epsilon) + b_neg
        c_n = g.gradient(R, s)
        return (self.alpha * c_p + self.beta * c_n) * x

    # Conv2D backpropogation 
    def relprop_conv(self, x, r, w, b, strides=(1, 1, 1, 1), padding='SAME'):
        b_pos = tf.maximum(b, 0)
        b_neg = tf.minimum(b, 0)
        if len(strides) == 2:
            strides = (1, strides[0], strides[1], 1)
        w_pos = tf.maximum(w, 0)
        w_neg = tf.minimum(w, 0)
        z = tf.nn.conv2d(x, w_pos, strides, padding) + self.epsilon + b_pos
        s = r / z
        c_p = tf.compat.v1.nn.conv2d_backprop_input(tf.shape(x), w_pos, s, strides, padding)

        
        z = tf.nn.conv2d(x, w_neg, strides, padding) + self.epsilon + b_neg
        s = r / z
        c_n = tf.compat.v1.nn.conv2d_backprop_input(tf.shape(x), w_neg, s,
                strides, padding)

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
        for my_layer in self.model.layers[-1::-1]:
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
                    my_layer.trainable_weights[0], my_layer.trainable_weights[1],
                    output_relevances[out_layer_name])
            elif 'conv2d' in my_layer.name:       
                output_relevances[my_layer.name.split("/")[0]] = self.relprop_conv(
                    x_layers[inp_layer_name],
                    output_relevances[out_layer_name],
                    my_layer.trainable_weights[0], my_layer.trainable_weights[1],
                    strides=my_layer.strides, 
                    padding=my_layer.padding.upper())
            elif 'batch_normalization' in my_layer.name:
                output_relevances[my_layer.name.split("/")[0]] = self.relprop_batchnorm(
                        x_layers[inp_layer_name], 
                        output_relevances[out_layer_name],
                        my_layer.trainable_weights[0], my_layer.trainable_weights[1])
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

if __name__ == "__main__":
    hour = 0
    my_model = tf.keras.models.load_model(
        '/lcrc/group/earthscience/rjackson/opencrums/models/classifier-mass-only-%d-nodes-2conv-' % int(sys.argv[1]))

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
        cat_number = air_now_df['CategoryNumber'].values[ind]
        if cat_number > 2:
            cat_number = np.nan
        return cat_number
    lat = 0
    lon = 0
    lat_slice = slice(25, 35)
    lon_slice = slice(-100, -90)
    # Load MERRA2 data from files
    def load_data(species):
        ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_extended/%sSMASS*.nc' % species,
                parallel=True)
        print(ds)
        times = np.array(list(map(pd.to_datetime, ds.time.values)))
        try:
            x = ds["%sSMASS25" % species].values
        except KeyError:
            x = ds["%sSMASS" % species].values
        lat = ds["lat"].values
        lon = ds["lon"].values
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
        times = times[where_valid]
        print(times.shape, classification.shape)
        y = np.concatenate([tf.one_hot(classification, 2).numpy(), times[:, np.newaxis]], axis=1)
        x_dataset_train = {'input_%ssmass' % species.lower(): x}

        return x_dataset_train, y[:,:2], y[:,2]

    x_ds_train = {}
    x_ds_test = {}
    y_train = []
    y_test = []
    species_list = ['SS', 'SO4', 'OC','DU', 'BC']
    for species in species_list:
        x_ds_train1, y_train, t_train = load_data(species)
        x_ds_train.update(x_ds_train1)

    for key in x_ds_train.keys():
        x_ds_train[key] = x_ds_train[key][:, :, :, np.newaxis]

    y_predict = my_model.predict(x_ds_train)
    y_predicts = y_predict.argmax(axis=1)

    def slice_input_ds(x: dict, y, y_train, t, start, end):
        x_new = {}
        for key in x.keys():
            x_new[key] = x[key][start:end, :, :, :].copy()
        y_new = y[start:end, :].copy()
        y_t = y_train[start:end, :].copy()
        t_test = t[start:end].copy()
        return x_new, y_new, y_t, t_test
    lrp = LayerwiseRelevancePropogation(my_model)
    rel_out = '/lcrc/group/earthscience/rjackson/opencrums/scripts/relevances_surface'

    ds = xr.open_dataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_extended/DUCMASS2010.nc').sortby('time').sel(lat=lat_slice, lon=lon_slice)
    lat = ds.lat.values
    lon = ds.lon.values
    num_times = 100
    for i in range(0, y_predict.shape[0], num_times):
        print(i + num_times)
        x_slice, y_slice, yt_slice, t_slice = slice_input_ds(x_ds_train, y_predict, y_train, t_train, i, i + num_times)
        relevance = lrp.calc_relevance(x_slice, y_slice)
        out_ds = {}
        out_ds['time'] = xr.DataArray(t_slice, dims=['time'])
        out_ds['aqi'] = xr.DataArray(yt_slice.argmax(axis=1), dims=['time'])
        means = []
        for key in relevance.keys():
            if "input_" in key:
                arr = relevance[key].numpy()

                times = out_ds["time"]
                ind_sort = np.argsort(times)
                arr = arr[ind_sort, 1:-1, 1:-1, :]
                times = times[ind_sort]
                out_ds["rel_" + key[6:]] = xr.DataArray(
                    np.squeeze(arr), dims=('time', 'lat', 'lon'))
            

    
        hist_bins = np.linspace(0, 100, 1000)
        for j in range(num_times):
            if i + j >= y_predict.shape[0]:
                continue
            relevance_arr = []
            for key in relevance.keys():
                if "input_" in key:
                    relevance_arr.append(out_ds["rel_" + key[6:]].values[j, :, :].flatten())
            # If we have NaNs, RLP is failing, we can't make a decision
            relevance_arr = np.stack(relevance_arr)
            rel_max = np.max(relevance_arr)
            if np.isnan(rel_max):
                print("NaNs detected skipping %d" % j)
                continue

            for key in relevance.keys():
                if "input_" in key:
                    #cdf = interp1d(bins[1:], cdfs, fill_value=(0, 1), bounds_error=False)
                    out_ds["rel_" + key[6:]][j] = out_ds["rel_" + key[6:]][j] / rel_max
                    # Check for numerical instability

        relevance_arr = np.concatenate(relevance_arr)
        out_ds['time'] = times[ind_sort]
        labels = relevance['output'].numpy().argmax(axis=1)
        labels = labels[np.array(ind_sort)]
        out_ds['label'] = ('time', labels)
        out_ds['lat'] = ('lat', np.squeeze(lat[1:-1]))
        out_ds['lon'] = ('lon', np.squeeze(lon[1:-1]))
        out_ds = xr.Dataset(out_ds)
        out_ds.to_netcdf(os.path.join(rel_out, 'relevance%d.nc' % i))
        out_ds.close()

    out_ds = xr.open_mfdataset(rel_out + "/*.nc", parallel=True)
    out_ds.to_netcdf('relevances_mass_only_%dnodes.nc' % int(sys.argv[1]))
