import tensorflow as tf
import sys
import ray
import xarray as xr
import numpy as np

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Add, ReLU, Conv2DTranspose, Dense, Dropout, MaxPooling3D, UpSampling3D
from tensorflow.keras.layers import TimeDistributed, LSTM, RepeatVector, ConvLSTM2D, Concatenate, Layer
from tensorflow.keras.applications import VGG19
from tensorflow.distribute import MirroredStrategy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow import keras
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

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae_encdec(input_shape):
    latent_dim = 2

    encoder_inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    
    new_shape = (int(input_shape[0]/4), int(input_shape[1]/4), input_shape[2])
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(new_shape[0] * new_shape[1] * 128, activation="relu")(latent_inputs)
    x = Reshape((new_shape[0], new_shape[1], 128))(x)
    x = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = Conv2DTranspose(
            input_shape[2], 3, activation="sigmoid", padding="same")(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    return encoder, decoder

class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data, reconstruction), axis=(1, 2)
                )
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

code = 'HOU'
if code == 'HOU':
    ax_extent = [-105, -85, 25, 35]
elif code == 'SEUS':
    ax_extent = [-90, -75, 30, 37.5]

def load_data(num_timesteps):
    ds = xr.open_mfdataset(
            '/lcrc/group/earthscience/rjackson/era5-preprocessed/*.nc')
    ds.load()
    ds = ds.sortby('time')
    x = np.squeeze(ds["u"].values)
    time = ds["time"].values
    old_shape = x.shape
     
    # Make sure that we can divide our dataset into the given intervals
    y = np.squeeze(ds["v"].values)
    which_lats = np.where(np.logical_and(ds["latitude"].values >= ax_extent[2], 
        ds["latitude"].values <= ax_extent[3]))[0]
    which_lons = np.where(np.logical_and(ds["longitude"].values >= ax_extent[0],
        ds["longitude"].values <= ax_extent[1]))[0]
    x = x[:,which_lats[0]:(which_lats[0]+96),which_lons[0]:(which_lons[0]+96)]
    y = y[:,which_lats[0]:(which_lats[0]+96),which_lons[0]:(which_lons[0]+96)]
    print(x.min())
    print(x.max())
    #x[~np.isfinite(x)] = 0
    #y[~np.isfinite(y)] = 0
    old_shape = y.shape
    x = np.stack([x, y], axis=-1)
    shape = x.shape
    x = tf.data.Dataset.from_tensor_slices(x) 
    ds.close()
    del ds, y
    return x, shape, time


def run(config: dict):
    x_ds, shape, time = load_data(1)
    encoder, decoder = build_vae_encdec(shape[1:])
    model = VAE(encoder, decoder)
    x_ds = x_ds.batch(config["batch_size"])
    model.compile(optimizer=Adam(lr=config["learning_rate"]))
    model.fit(x_ds, epochs=config["num_epochs"])
    z_mean, z_std, _ = model.encoder.predict(x_ds)
    
    out_ds = xr.Dataset({'time': (['time'], time), 
        'z_mean': (['time', 'encoding'], z_mean),
        'z_std': (['time', 'encoding'], z_std)})
    out_ds.to_netcdf('encodings_era5_winds.nc')
    model.save('../models/default_vae_era5_winds') 
    

default_config = {
        "num_epochs": 500,
        "num_channels": 232,
        "learning_rate": 1e-3,
        "batch_size": 16}

run(default_config)
print(f"MSE Default Configuration:  {objective_default:.3f}")


