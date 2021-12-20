import tensorflow as tf
import sys
import ray
import xarray as xr
import numpy as np

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Add, ReLU, Conv2DTranspose, Dense, Dropout
from tensorflow.keras.layers import TimeDistributed, LSTM, RepeatVector
from tensorflow.distribute import MirroredStrategy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from sklearn.preprocessing import StandardScaler
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


def multiencoder_model(shape, var, the_dict):
    width = shape[3]
    height = shape[2]
    inp_layer = Input(shape=(the_dict['num_timesteps'], height, width), name=var)
    lstm_input = Reshape(target_shape=(the_dict['num_timesteps'], height*width))(inp_layer)
    encoding = LSTM(the_dict['num_channels'], activation=the_dict['activation'],
            input_shape=(the_dict['num_timesteps'], height * width))(lstm_input)
    repeat = RepeatVector(the_dict['num_timesteps'])(encoding)
    decoder = LSTM(the_dict['num_channels'], activation=the_dict['activation'],
            return_sequences=True)(repeat)
    output = TimeDistributed(Dense(height * width))(decoder)
    output = Reshape(target_shape=(the_dict['num_timesteps'], height, width))(output)
    return Model(inp_layer, output)

tfrecords_path = '/lcrc/group/earthscience/rjackson/MERRA2/tfrecords/*.tfrecord'
normalizers_path = '/lcrc/group/earthscience/rjackson/opencrums/models/normalizers/'
def load_data(num_timesteps):
    ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/DUCMASS*.nc')
    ds = ds.sortby('time')
    print(ds)
    x = ds["DUCMASS"].values
    old_shape = x.shape
    # Make sure that we can divide our dataset into the given intervals
    x = x[:(int(old_shape[0] / num_timesteps) * num_timesteps), :, :]
    old_shape = x.shape
    scaler = StandardScaler()
    scaler.fit(np.reshape(x, (int(old_shape[0]), old_shape[1] * old_shape[2])))
    x = scaler.transform(np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
    new_shape = (int(old_shape[0]/num_timesteps), num_timesteps,
        old_shape[1], old_shape[2])
    x = np.reshape(x, new_shape)
    x_dataset = tf.data.Dataset.from_tensor_slices((x, x))
    shape = x.shape
    return x_dataset, shape


def run(config: dict):
    x_ds, shape = load_data(config['num_timesteps'])
    model = multiencoder_model(shape, sys.argv[1], config)
    model.summary()
    x_ds = x_ds.batch(config["batch_size"])
    model.compile(optimizer=Adam(lr=config["learning_rate"]),
        loss="mean_squared_error", metrics=['mse'])
    history = model.fit(x_ds, epochs=config["num_epochs"])
    return history.history["mse"][-1]

default_config = {
        "num_epochs": 100,
        "num_channels": 16,
        "learning_rate": 1e-3,
        "num_timesteps": 4,
        "activation": "relu",
        "batch_size": 16}

if not ray.is_initialized():
    ray.init(num_cpus=8, num_gpus=8, log_to_driver=False)
    run_default = ray.remote(num_cpus=1, num_gpus=1)(run)
    objective_default = ray.get(run_default.remote(default_config))
#run(default_config)
print(f"MSE Default Configuration:  {objective_default:.3f}")

problem = HpProblem()
problem.add_hyperparameter((20, 500), "num_epochs")
problem.add_hyperparameter((4, 20), "num_timesteps")
problem.add_hyperparameter((4, 256, "log-uniform"), "num_channels")
# Categorical hyperparameter (sampled with uniform prior)
ACTIVATIONS = [
    "elu", "gelu", "hard_sigmoid", "linear", "relu", "selu",
    "sigmoid", "softplus", "softsign", "swish", "tanh",
]
problem.add_hyperparameter(ACTIVATIONS, "activation")
problem.add_hyperparameter((1e-5, 1e-2, "log-uniform"), "learning_rate")
problem.add_hyperparameter((8, 256, "log-uniform"), "batch_size")

problem.add_starting_point(**default_config)

method_kwargs = {
        "num_cpus": 8,
        "num_cpus_per_task": 1,
        "callbacks": [LoggerCallback()]
    }

method_kwargs["num_cpus"] = 8
method_kwargs["num_gpus"] = 8
method_kwargs["num_cpus_per_task"] = 1
method_kwargs["num_gpus_per_task"] = 1

evaluator = Evaluator.create(run, method="ray", method_kwargs=method_kwargs)
search = AMBS(problem, evaluator)
results = search.search(100)
results.to_csv('hpsearch_results_lstm.csv')
