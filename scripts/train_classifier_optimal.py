import tensorflow as tf
import sys
import ray
import xarray as xr
import numpy as np

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Add, ReLU, Conv2DTranspose, Dense
from tensorflow.distribute import MirroredStrategy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
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


def classifier_model(shape, var, the_dict):
    width = shape[2]
    height = shape[1]
    inp_layer = Input(shape=(height, width, 1), name=var)
    for i in range(the_dict['num_layers']):
        conv2d_1 = Conv2D(the_dict['num_channels'],
                (2, 2), activation=the_dict['activation'], padding='same',
                kernel_initializer='he_normal')(inp_layer)
        mpool_1 = MaxPooling2D((2, 2))(conv2d_1)
    
    flat_1 = Flatten()(mpool_1)
    for i in range(the_dict['num_dense_layers']):
        flat_1 = Dense(the_dict['num_dense_nodes'])(flat_1)
    output = Dense(4, activation="softmax", name="class")(flat_1)
    return Model(inp_layer, output)

tfrecords_path = '/lcrc/group/earthscience/rjackson/MERRA2/tfrecords/*.tfrecord'
normalizers_path = '/lcrc/group/earthscience/rjackson/opencrums/models/normalizers/'
def load_data():
    ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/DUCMASS*.nc')
    print(ds)
    x = ds["DUCMASS"].values
    old_shape = x.shape
    scaler = StandardScaler()
    scaler.fit(np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
    x = scaler.transform(np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
    x = np.reshape(x, old_shape)
    class_ds = xr.open_dataset('classification_dust.nc')
    y = tf.one_hot(class_ds.classification.values, 4).numpy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    x_dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    x_dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    shape = x.shape
    return x_dataset_train, x_dataset_test, shape


def run(config: dict):
    x_ds_train, x_ds_test, shape = load_data()
    model = classifier_model(shape, sys.argv[1], config)
    x_ds_train = x_ds_train.batch(config["batch_size"])
    x_ds_test = x_ds_test.batch(config["batch_size"])
    model.compile(optimizer=Adam(lr=config["learning_rate"]),
        loss="categorical_crossentropy", metrics=['acc'])
    history = model.fit(x_ds_train, validation_data=x_ds_test, epochs=config["num_epochs"])
    return history.history["val_acc"][-1]

default_config = {
        "num_epochs": 100,
        "num_channels": 16,
        "learning_rate": 1e-3,
        "num_dense_nodes": 8,
        "num_dense_layers": 2,
        "activation": "relu",
        "batch_size": 16,
        "num_layers": 2}

if not ray.is_initialized():
    ray.init(num_cpus=8, num_gpus=8, log_to_driver=False)

run_default = ray.remote(num_cpus=1, num_gpus=1)(run)
objective_default = ray.get(run_default.remote(default_config))

print(f"MSE Default Configuration:  {objective_default:.3f}")

problem = HpProblem()
problem.add_hyperparameter((20, 400), "num_epochs")
problem.add_hyperparameter((8, 512, "log-uniform"), "num_dense_nodes")
problem.add_hyperparameter((1, 2), "num_layers")
problem.add_hyperparameter((1, 8), "num_dense_layers")
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

method_kwargs["num_cpus"] = 16
method_kwargs["num_gpus"] = 16
method_kwargs["num_cpus_per_task"] = 1
method_kwargs["num_gpus_per_task"] = 1

evaluator = Evaluator.create(run, method="ray", method_kwargs=method_kwargs)
search = AMBS(problem, evaluator)
results = search.search(500)
results.to_csv('hpsearch_results_classifierDCMASS.csv')
