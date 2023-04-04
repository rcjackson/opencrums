import pandas as pd
import numpy as np
from glob import glob

air_now_data = glob('/lcrc/group/earthscience/rjackson/epa_air_now/*.csv')
air_now_df = pd.concat(map(pd.read_csv, air_now_data))
air_now_df['datetime'] = pd.to_datetime(air_now_df['DateObserved'] + ' 00:00:00')
air_now_df = air_now_df.set_index('datetime')
air_now_df = air_now_df.sort_index()

AQI = air_now_df.AQI.values[air_now_df.ParameterName.values == "PM2.5"]
for percentiles in [20, 40, 60, 80]:
    print(np.nanpercentile(AQI, percentiles))
print(air_now_df)
