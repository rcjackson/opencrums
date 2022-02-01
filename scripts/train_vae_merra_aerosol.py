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
            '/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/DUCMASS*.nc')
    print(ds)
    x = ds["DUCMASS"].values
    old_shape = x.shape
    scaler = StandardScaler()
    scaler.fit(np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
    x = scaler.transform(np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
    x = np.reshape(x, old_shape)
    ds.close()
    ds = xr.open_mfdataset(
            '/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/DUFLUXU*.nc')
    print(ds)
    y = ds["DUFLUXU"].values
    time = ds["time"].values
    print(time.shape)
    old_shape = x.shape
    scaler = StandardScaler()
    scaler.fit(np.reshape(y, (old_shape[0], old_shape[1] * old_shape[2])))
    y = scaler.transform(
            np.reshape(y, (old_shape[0], old_shape[1] * old_shape[2])))
    y = np.reshape(y, old_shape)
    ds.close()
    ds = xr.open_mfdataset(
            '/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/DUFLUXV*.nc')
    z = ds["DUFLUXV"].values
    old_shape = x.shape
    scaler = StandardScaler()
    scaler.fit(np.reshape(z, (old_shape[0], old_shape[1] * old_shape[2])))
    z = scaler.transform(
            np.reshape(z, (old_shape[0], old_shape[1] * old_shape[2])))
    z = np.reshape(z, old_shape)
    ds.close()
    x = np.stack([x, y, z], axis=-1)
    shape = x.shape
    x_array = x
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
    out_ds.to_netcdf('dust_encodings_aerosol.nc')
    model.save('../models/default_vae_aerosol_%s' % sys.argv[1]) 
    return history.history["mse"][-1]

default_config = {
        "num_epochs": 500,
        "num_channels": 232,
        "learning_rate": 1e-2,
        "batch_size": 16}

objective_default = run(default_config)
print(f"MSE Default Configuration:  {objective_default:.3f}")


