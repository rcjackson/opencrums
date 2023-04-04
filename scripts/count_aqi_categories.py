import pandas as pd
import numpy as np

from glob import glob

air_now_data = glob('/lcrc/group/earthscience/rjackson/epa_air_now/*.csv')
air_now_df = pd.concat(map(pd.read_csv, air_now_data))
air_now_df['datetime'] = pd.to_datetime(air_now_df['DateObserved'] + ' 00:00:00')
air_now_df = air_now_df.set_index('datetime')
air_now_df = air_now_df.sort_index()
air_now_df = air_now_df[air_now_df['ParameterName'] == "PM2.5"]
print(air_now_df['CategoryNumber'].values.min())

hist, bins = np.histogram(air_now_df['CategoryNumber'].values, bins=np.arange(0.5, 6, 1))
print(hist)

