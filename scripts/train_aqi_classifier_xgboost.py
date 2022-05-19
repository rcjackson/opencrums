import pandas as pd
import xarray as xr
import glob as glob
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from datetime import timedelta, datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import DMatrix, train, XGBClassifier

air_now_data = glob.glob('/lcrc/group/earthscience/rjackson/epa_air_now/*.csv')
air_now_df = pd.concat(map(pd.read_csv, air_now_data))
air_now_df['datetime'] = pd.to_datetime(air_now_df['DateObserved'] + ' 00:00:00')
air_now_df = air_now_df.set_index('datetime')
air_now_df = air_now_df.sort_index()
print(air_now_df['CategoryNumber'].values.min())

out_png_path = '/lcrc/group/earthscience/rjackson/opencrums/xgb_importances/'

def get_air_now_label(time):
    if np.min(np.abs((air_now_df.index - time))) > timedelta(days=1):
        return np.nan
    ind = np.argmin(np.abs(air_now_df.index - time))
    return air_now_df['CategoryNumber'].values[ind]

# Get lats, lons for plotting
ds = xr.open_mfdataset(
            '/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/DUCMASS*.nc')
print(ds.time)
x = ds["DUCMASS"].values
lon = ds["lon"].values
lat = ds["lat"].values
ds.close()

def load_data(species):
    ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/%sCMASS*.nc' % species).sortby('time')
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
    ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/%sFLUXU*.nc' % inp).sortby('time')
    x2 = ds["%sFLUXU" % inp].values
    scaler = StandardScaler()
    scaler.fit(np.reshape(x2, (old_shape[0], old_shape[1] * old_shape[2])))
    x2 = scaler.transform(
            np.reshape(x2, (old_shape[0], old_shape[1] * old_shape[2])))
    x2 = np.reshape(x2, old_shape)
    inputs[:, :, :, 1] = x2
    ds.close()
    ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/%sFLUXV*.nc' % inp).sortby('time')
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
    classification = classification[where_valid] 
    #enc = OneHotEncoder(handle_unknown='ignore')
    
    #y = enc.fit_transform(classification.reshape(-1, 1)).toarray()
    y = classification
    #y = y.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(
            inputs, y, test_size=0.20, random_state=3)
    shape = inputs.shape
    x_dataset_train = {'input_%sMASS' % species: np.squeeze(x_train[:, :, :, 0]),
            'input_%sFLUXU' % inp: np.squeeze(x_train[:, :, :, 1]),
            'input_%sFLUXV' % inp: np.squeeze(x_train[:, :, :, 2])}
    x_dataset_test = {'input_%sMASS' % species: np.squeeze(x_test[:, :, :, 0]),
            'input_%sFLUXU' % inp: np.squeeze(x_test[:, :, :, 1]),
            'input_%sFLUXV' % inp: np.squeeze(x_test[:, :, :, 2])}

    return x_dataset_train, x_dataset_test, y_train, y_test, shape


x_ds_train = {}
x_ds_test = {}
y_train = []
y_test = []
species_list = ['SS', 'SO4', 'SO2', 'OC','DU', 'DMS', 'BC']
for species in species_list:
    print(species)
    x_ds_train1, x_ds_test1, y_train, y_test, shape = load_data(species)
    x_ds_train.update(x_ds_train1)
    x_ds_test.update(x_ds_test1)
input_keys = [x for x in x_ds_test.keys()]
x_ds_test = np.stack([x for x in x_ds_test.values()], axis=-1)
x_ds_train = np.stack([x for x in x_ds_train.values()], axis=-1)
x_ds_test = np.reshape(x_ds_test, (x_ds_test.shape[0], 
    np.prod(x_ds_test.shape[1:]))).astype(np.float32)
x_ds_train = np.reshape(x_ds_train, (x_ds_train.shape[0],
    np.prod(x_ds_train.shape[1:]))).astype(np.float32)
print(x_ds_train.shape)
print(y_train.shape)
    #train_set = DMatrix(x_ds_train, label=y_train.astype(np.float32))
    #test_set = DMatrix(x_ds_test, label=y_test.astype(np.float32))
#eval_set = [(x_ds_test, y_test)]
#for i in range(x_ds_test.shape[0]):
#    eval_set.append((x_ds_test[i], np.array(y_test[i])))
eval_set = (x_ds_test, y_test)
evals_result = {}

def run(config: dict):
    #bst = RayXGBClassifier(*config)
    #bst.fit(x_ds_train, y_train, verbose=True,
    #        eval_set=eval_set, ray_params=ray_params)
    bst = XGBClassifier(tree_method="gpu_hist",
            objective="multi:softmax",
            eval_metric="merror",
            eta=1e-3,
            subsample=0.7,
            num_class=5,
            max_depth=15,
            early_stopping_rounds=50,
            n_jobs=1)
    #bst.set_param("nthread", 4)
    print(bst.get_xgb_params())
    print(y_train)
    print(x_ds_train.shape)
    bst.fit(x_ds_train, y_train, verbose=True)
    #bst.save_model("model.xgb")
    # AQI classes inbalanced, need weights
    return bst

config = {
        "tree_method": "gpu_hist",
        "objective": "multi:softmax",
        "eval_metric": "merror",
        "eta": 1e-3,
        "subsample": 0.7,
        "num_class": 5,
        "max_depth": 15,
        "early_stopping_rounds": 50,
        "n_jobs": 1,
        "gpu_id": 1}

#run(config)
# Specify the hyperparameter search space.
#config = {
#    "tree_method": "approx",
#    "objective": "multi:softmax",
#    "eval_metric": ["merror"],
#    "eta": tune.loguniform(1e-4, 1e-1),
#    "num_class": 5,
#    "subsample": tune.uniform(0.5, 1.0),
#    "max_depth": tune.randint(1, 9)
#}
result = run(config)

feature_importance = result.feature_importances_
#feature_importance = np.array([x for x in feature.values()])
#print(feature_importance)
feature_importance = np.reshape(feature_importance,
        (len(input_keys), shape[1], shape[2]))
print(feature_importance[0:10])
i = 0
for key in input_keys:
    r = feature_importance[i] 
    fig, ax = plt.subplots(1, 1, figsize=(8, 8),
            subplot_kw=dict(projection=ccrs.PlateCarree()))
    c = ax.contourf(lon, lat, r,
        cmap='coolwarm', levels=np.logspace(-6, -3, 100))
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.set_title(key)
    ax.set_xlabel('Latitiude')
    ax.set_ylabel('Longitude')
    plt.colorbar(c, label='Feature importance')
    fig.savefig(out_png_path + '/relevance_%s.png' % key,
            bbox_inches='tight', dpi=300)
    i = i + 1
    plt.close(fig)
    
#print(result)
# Make sure to use the `get_tune_resources` method to set the `resources_per_trial`
#ray_params = RayParams(num_actors=4, cpus_per_actor=16, gpus_per_actor=1)
#analysis = tune.run(
#    run,
#    config=config,
#    metric="eval-merror",
#    mode="min",
#    num_samples=64,
#    log_to_file=True,
#    reuse_actors=True,
#    resources_per_trial=ray_params.get_tune_resources())

