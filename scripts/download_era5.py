import cdsapi
import sys
import os
import numpy as np
from datetime import datetime, timedelta
from dateutil import relativedelta 

out_path = '/lcrc/group/earthscience/rjackson/era5/'

def get_file(date):
    c = cdsapi.Client()
    year_str = '%04d' % date.year
    month_str = '%02d' % date.month
    days = np.arange(0, 7, 1) + date.day
    if date.month == 2:
        if date.year % 4 > 0:
            days = days[days < 29]
        else:
            days = days[days < 30]
    elif date.month == 4 or date.month == 6 or date.month == 9 or date.month == 1:
        days = days[days < 31]
    else:
        days = days[days < 32]

    day_array = [str(x) for x in days]
    out_file = out_path + '/%s/%s/%s%s%s-%sera5.grib' % (year_str, month_str,
            year_str, month_str, day_array[0], day_array[-1])
    head, tail = os.path.split(out_file)
    if not os.path.isdir(head):
        os.makedirs(head)
    if os.path.exists(out_file):
        return
    c.retrieve('reanalysis-era5-pressure-levels',
            {
            'product_type': 'reanalysis',
            'format': 'grib',
            'variable': [
                'divergence', 'fraction_of_cloud_cover', 'geopotential',
                'ozone_mass_mixing_ratio', 'potential_vorticity', 'relative_humidity',
                'specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content', 'specific_humidity',
                'specific_rain_water_content', 'specific_snow_water_content', 'temperature',
                'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
                'vorticity',
            ],
            'pressure_level': [
                '1', '2', '3', '5', '7', '10',
                '20', '30', '50', '70', '100', '125',
                '150', '175', '200', '225', '250', '300',
                '350', '400', '450', '500', '550', '600',
                '650', '700', '750', '775', '800', '825',
                '850', '875', '900', '925', '950', '975',
                '1000',]
            , 'year': year_str
            , 'month': month_str
            , 'day': day_array
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
    for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
#for year in [2017]:
#    for month in [6, 7, 8, 9]:
        if year == 2011 and month < 5:
            continue
        the_day = datetime(year, month, 1, 1, 1, 1)
        end_day = the_day + relativedelta.relativedelta(months=1)
        while the_day < end_day:
            print('Retrieving %s' % str(the_day))
            get_file(the_day)
            the_day = the_day + timedelta(days=7)

