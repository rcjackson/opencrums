import pandas as pd
import sys

year = int(sys.argv[1])
air_now_out = '/lcrc/group/earthscience/rjackson/epa_air_now_seus'

dates = pd.date_range(start='%d-01-01' % year, end='%d-12-31' % year, freq='1D')
for day in dates:
    day_format = day.strftime('%Y-%m-%dT%H-%M%S')
    print(day_format)
    try:
        csv = pd.read_csv('https://www.airnowapi.org/aq/observation/zipCode/historical/?format=text/csv&zipCode=35553&date=' + day_format + '&distance=75&API_KEY=6A36C701-CCE3-404B-BAF1-6DBF07B03D52')
        csv.to_csv(air_now_out + '/airnow%s.csv' % day.strftime('%Y%m%d'))
    except:
        continue

