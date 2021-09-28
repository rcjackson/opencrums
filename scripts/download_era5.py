import cdsapi
import sys
from datetime import datetime, timedelta
from dateutil 

out_path = '/lambda_stor/data/rjackson/era5/'

def get_file(date):
    c = cdsapi.Client()
    year_str = '%04d' % date.year
    month_str = '%02d' % date.month
    day_str = '%02d' % date.day
    out_file = out_path + '/%s/%s/%s%s%sera5.grib' % (year_str, month_str,
            year_str, month_str, day_str)
    c.retrieve('reanalysis-era5-pressure-levels',
            {
            'product_type': 'reanalysis',
            'format': 'grib',
            'variable': [
                'specific_humidity', 'temperature', 'u_component_of_wind',
                'v_component_of_wind', 'relative_humidity'],
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
            , 'day': day_str
            , 'time': ['00:00', '01:00', '02:00', '03:00',
                '04:00', '05:00', '06:00', '07:00',
                '08:00', '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00', '15:00',
                '16:00', '17:00', '18:00', '19:00',
                '20:00', '21:00', '22:00', '23:00']
            }, out_file)
    return

the_day = datetime(int(sys.argv[1]), int(sys.argv[2]), 1, 1, 1, 1)
end_day = the_day + timedelta(months=1)
while the_day < end_day:
    print('Retrieving %s' % str(the_day))
    get_file(the_day)
    the_day = the_day + timedelta(days=1)






