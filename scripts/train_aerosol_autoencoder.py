import tensorflow as tf
import xarray as xr
import glob as glob
import numpy as np
import sys
import pickle
import xbatcher
import matplotlib.pyplot as plt
import os

#import ray

from datetime import timedelta, datetime
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Add, ReLU, Conv2DTranspose, Dense, Dropout, BatchNormalization, Lambda, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xbatcher.loaders.keras import CustomTFDataset

species_list = ["DUCMASS", "DUSMASS", "DMSCMASS", "DMSSMASS", "DUFLUXU",
                "DUFLUXV", "SO4CMASS", "SO2CMASS", "SO4SMASS", "SO2SMASS",
                "OCCMASS", "OCSMASS", "OCFLUXU", "OCFLUXV", "SSFLUXU",
                "SSFLUXV", "SSCMASS", "SSSMASS", "BCCMASS", "BCSMASS",
                "BCFLUXU", "BCFLUXV"]


def preprocess_dataset(ds, species):
    x = ds[species].values
    min_gt0 = np.min(x[x > 0])
    x[x == 0] = min_gt0
    if not "FLUX" in species:
        x = np.log10(x)
    x = x - x.min()
    old_shape = x.shape
    #scaler = MinMaxScaler()
    #scaler.fit(np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
    #x = scaler.transform(
    #    np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
    #x = np.reshape(x, old_shape)
    
    x = x[:, :, :, np.newaxis]
    x = x / x.max() 
    print(x.min(), x.max())
    x = tf.image.resize(x, (64, 64)).numpy()
    ds[species] = xr.DataArray(x, dims=('time', 'x', 'y', 'channel'))
    return ds


def load_data(species):
    preproc = lambda x: preprocess_dataset(x, species)
    ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_extended/%s*.nc' % species,
                           preprocess=preproc).sortby('time')
    x_train, x_test = train_test_split(
            ds[species], test_size=0.20, random_state=3)
    x_test, x_validation = train_test_split(x_test, test_size=0.50,
            random_state=3)
    return x_train, x_test, x_validation


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


latent_dim = 16
def encoder_model(shape, the_dict, species):
    width = shape[2]
    height = shape[1]
    input_layer = Input(shape=(height, width, 1), name=species)
    mpool_1 = input_layer
    for i in range(the_dict['num_layers']):
        conv2d_1 = Conv2D(the_dict['num_channels'], (3, 3), activation='relu', padding='same')(mpool_1)
        conv2d_1 = MaxPooling2D((2, 2))(conv2d_1)
        mpool_1 = conv2d_1
    mpool_1 = Flatten()(mpool_1) 
    z_mean = Dense(latent_dim, name="z_mean")(mpool_1)
    z_log_var = Dense(latent_dim, name="z_log_var")(mpool_1)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(input_layer, [z_mean, z_log_var, z], name="encoder")
    return encoder

def decoder_model(shape, the_dict):
    latent_inputs = Input(shape=(latent_dim,))
    width = shape[2]
    height = shape[1]
    input1 = Dense(16 * 16 * the_dict['num_channels'])(latent_inputs)
    conv2d_1 = Reshape((16, 16, the_dict['num_channels']))(input1)
    for i in range(the_dict['num_layers']):
        conv2d_1 = Conv2DTranspose(the_dict['num_channels'], (3, 3), activation='relu', padding='same')(conv2d_1)
        conv2d_1 = UpSampling2D((2,2))(conv2d_1)
    output = Conv2DTranspose(1, (3, 3), padding='same', activation='sigmoid')(conv2d_1)
    return Model(latent_inputs, output)


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data[0])
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(
                        data[0], reconstruction))
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def call(self, inputs):
        z_mean, z_std, z = self.encoder(inputs)
        return self.decoder(z)

def run(config: dict):
    species = species_list[int(sys.argv[1])]
    x_ds_train, x_ds_test, x_ds_validation = load_data(species)
    x_ds_train_gen = x_ds_train.batch.generator(
        input_dims={'x': 64, 'y': 64, 'channel': 1},
        batch_dims={'time': config["batch_size"]})
    x_ds_test_gen = x_ds_test.batch.generator(
        input_dims={'x': 64, 'y': 64, 'channel': 1},
        batch_dims={'time': config["batch_size"]})
    x_ds_valid_gen = x_ds_validation.batch.generator(
        input_dims={'x': 64, 'y': 64, 'channel': 1},
        batch_dims={'time': config["batch_size"]})

    shape = x_ds_train.shape
    encoder = encoder_model(shape, config, species)
    decoder = decoder_model(shape, config)
    model = VAE(encoder, decoder)
    encoder.summary()
    decoder.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]))
    dataset_train = CustomTFDataset(x_ds_train_gen, x_ds_train_gen)
    dataset_test = CustomTFDataset(x_ds_test_gen, x_ds_test_gen)
    dataset_valid = CustomTFDataset(x_ds_valid_gen, x_ds_valid_gen)
    checkpoint_file_path = '../models/autoencoder-checkpoint-%s' % species
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_weights_only=True,
            monitor='reconstruction_loss', filepath=checkpoint_file_path,
            save_best_only=True, mode='min')
    if len(glob.glob(checkpoint_file_path + "*")) > 0:
        print("Loading %s" % checkpoint_file_path)
        model.load_weights(checkpoint_file_path)
    history = model.fit(
            dataset_train, 
            epochs=config["num_epochs"], callbacks=[EarlyStopping(patience=100,
            monitor="reconstruction_loss"), checkpoint])
    model.save('../models/autoencoder-large-%s' % species)
    # Plot quicklooks of trained result
    if not os.path.exists('../models/test_output/%s' % species):
        os.makedirs('../models/test_output/%s' % species)

    i = 0 
    for test_batch, y_batch in dataset_test:
        fig, ax = plt.subplots(config["batch_size"], 2, figsize=(5, 20))
        prediction = model.predict(test_batch)
        for j in range(config["batch_size"]):
            ax[j, 0].pcolormesh(test_batch[j].numpy().squeeze(), vmin=0, vmax=1, cmap='Greys_r')
            ax[j, 0].axis('off')
            ax[j, 1].pcolormesh(prediction[j].squeeze(), vmin=0, vmax=1, cmap='Greys_r')
            ax[j, 1].axis('off')
        fig.savefig('../models/test_output/%s/%d.png' % (species, i), bbox_inches='tight')
        plt.close(fig)
        del fig
        i += 1
        if i == 100:
            break

    return history.history

default_config = {
        "num_epochs": 10000,
        "num_channels": 512,
        "learning_rate": 0.001,
        "num_layers": 2,
        "activation": "relu",
        "threshold": 0, 
        "batch_size": 16}

run(default_config)
