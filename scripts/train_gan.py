import tensorflow as tf
import time
import xarray as xr
import glob as glob
import numpy as np
import sys
import pickle
import xbatcher
import os

#import ray

from datetime import timedelta, datetime
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xbatcher.loaders.keras import CustomTFDataset

species_list = ["DUCMASS", "DUSMASS", "DMSCMASS", "DMSSMASS", "DUFLUXU",
                "DUFLUXV", "SO4CMASS", "SO2CMASS", "SO4SMASS", "SO2SMASS",
                "OCCMASS", "OCSMASS", "OCFLUXU", "OCFLUXV", "SSFLUXU",
                "SSFLUXV", "SSCMASS", "SSSMASS", "BCCMASS", "BCSMASS",
                "BCFLUXU", "BCFLUXV"]

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

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


def generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*8*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((4, 8, 256)))
    assert model.output_shape == (None, 4, 8, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 4, 8, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 16, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 16, 32, 1)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[16, 32, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

generator = generator_model()
discriminator = make_discriminator_model()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

BATCH_SIZE = 16
noise_dim = 100
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

seed = tf.random.normal([num_examples_to_generate, noise_dim])

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch[0])
            
def run(config: dict):
    species = "BCCMASS"
    x_ds_train, x_ds_test = load_data(species)

    x_ds_train_gen = x_ds_train.batch.generator(
        input_dims={'x': 16, 'y': 32, 'channel': 1},
        batch_dims={'time': config["batch_size"]})
    x_ds_test_gen = x_ds_test.batch.generator(
        input_dims={'x': 16, 'y': 32, 'channel': 1},
        batch_dims={'time': config["batch_size"]})
    shape = x_ds_train.shape
    print(x_ds_train_gen[1])
    dataset_train = CustomTFDataset(x_ds_train_gen, x_ds_train_gen)
    dataset_test = CustomTFDataset(x_ds_test_gen, x_ds_train_gen)
    train(dataset_train, 50)
    #history = model.fit(
    #    dataset_train, validation_data=dataset_test,
    #    epochs=config["num_epochs"], callbacks=[EarlyStopping(patience=100,
    #    monitor="val_loss")])
    #model.save('../models/autoencoder-%s-lag-%d' % (variable, lag))
    discriminator.save('../models/discriminator-%s' % variable)
    generator.save('../model/generator-%s' % variable)
default_config = {
        "num_epochs": 10000,
        "num_channels": 512,
        "learning_rate": 0.01,
        "num_layers": 4,
        "activation": "relu",
        "batch_size": 16}

run(default_config)
