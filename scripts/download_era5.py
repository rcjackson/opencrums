import cdsapi
import sys
import os
import numpy as np
from datetime import datetime, timedelta
from dateutil import relativedelta 

out_path = '/lcrc/group/earthscience/rjackson/era5-surface/'

def get_file(date):
    c = cdsapi.Client()
    year_str = '%04d' % date

    out_file = out_path + '/%sera5.grib' % (year_str)
    head, tail = os.path.split(out_file)
    if not os.path.isdir(head):
        os.makedirs(head)
    if os.path.exists(out_file):
        return
    c.retrieve('reanalysis-era5-single-levels',
            {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
            '100m_u_component_of_wind', '100m_v_component_of_wind', '10m_u_component_of_wind',
            '10m_v_component_of_wind', '2m_dewpoint_temperature', '2m_temperature',
            'mean_sea_level_pressure', 'sea_surface_temperature', 'surface_pressure', 'convective_available_potential_energy', 'convective_inhibition'
        ]
            , 'year': year_str
            , 'time': ['00:00', '01:00', '02:00', '03:00',
                '04:00', '05:00', '06:00', '07:00',
                '08:00', '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00', '15:00',
                '16:00', '17:00', '18:00', '19:00',
                '20:00', '21:00', '22:00', '23:00'],
            'area': [20, -130, 45, -60],
            }, out_file)
    return

for year in [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]:
    print('Retrieving %s' % str(year))
    get_file(year)

