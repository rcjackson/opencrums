import pysplit

working_dir = '/lcrc/project/land_atmos_modeling/caghili/Hytraj_ParentDir/HyTraj_1000m/amd_hysplit/working/'
storage_dir = '/lcrc/group/earthscience/rjackson/opencrums/trajectories'
meteo_dir = '/lcrc/group/earthscience/rjackson/gdas'

basename = 'houston'
years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
months = [6, 7, 8, 9]
hours = [0]
altitudes = [300.]
location = (29.7604, -95.3698)
runtime = -120

pysplit.generate_bulktraj(basename, working_dir, storage_dir, meteo_dir,
                          years, months, hours, altitudes, location, runtime,
                          monthslice=slice(0, 32, 1), get_reverse=True,
                          get_clipped=True, hysplit='/lcrc/project/land_atmos_modeling/caghili/Hytraj_ParentDir/HyTraj_1000m/amd_hysplit/hyts_std')

