#!/usr/bin/env python
# coding: utf-8

# This Notebook documents the creation of SQL database melbourne_db.

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
import datetime as dt

dbname = 'melbourne_db'
username = 'emilyvoytek'
pswd = 'founded1835'

## 'engine' is a connection to a database
engine = create_engine('postgresql://%s:%s@localhost/%s'%(username,pswd,dbname))


# LOAD THE HISTORIC PED DATA
ped_historic = pd.read_csv('./data/Pedestrian_Counting_System___2009_to_Present__counts_per_hour_.csv')
ped_historic.columns=(['id', 'date_time', 'year', 'month', 'mdate', 'day', 'time', 'sensor_id',
       'sensor_name', 'hourly_counts'])
ped_historic['hourly_counts'] = ped_historic['hourly_counts'].replace(',','', regex=True).astype(float)


# ATTENTION: THIS TAKES FOREVER (~5-10 min) TO RUN
ped_historic.to_sql('data_ped_historic', engine, if_exists='replace')


# LOAD THE STATION INFORMATION
ped_stations = pd.read_csv('./data/Pedestrian_Counting_System_Sensor_Locations.csv')
ped_stations['installation_date'] = pd.to_datetime(ped_stations['installation_date'])

ped_stations.set_index("sensor_id",inplace=True)
ped_stations = ped_stations.sort_index()
ped_stations.to_sql('data_ped_stations', engine, if_exists='replace')
ped_stations


# LOAD HISTORIC WEATHER DATA
df_rain = pd.read_csv('./data/IDCJAC0009_086338_1800_Data.csv')
df_rain.columns = ['product_code', 'station_number', 'year', 'month', 'day', 'rainfall_mm','days_measured', 'quality']
df_rain['datetime']=pd.to_datetime(df_rain[['year','month', 'day']])
df_rain = df_rain.set_index(pd.to_datetime(df_rain['datetime']))
#df_rain.to_sql('data_rain', engine, if_exists='replace')
#
df_temp_max = pd.read_csv('./data/IDCJAC0010_086338_1800_Data.csv')
df_temp_max.columns = ['product_code', 'station_number', 'year', 'month', 'day', 'maximum_temperature_C',
       'days_measured', 'quality']
df_temp_max['datetime']=pd.to_datetime(df_temp_max[['year','month', 'day']])
df_temp_max = df_temp_max.set_index(pd.to_datetime(df_temp_max['datetime']))
#df_temp_max.to_sql('data_max_temp', engine, if_exists='replace')
#
df_temp_min = pd.read_csv('./data/IDCJAC0011_086338_1800_Data.csv')
df_temp_min.columns = ['product_code', 'station_Number', 'year', 'month', 'day', 'minimum_temperature_C',
       'days_measured', 'quality']
df_temp_min['datetime']=pd.to_datetime(df_temp_min[['year','month', 'day']])
df_temp_min = df_temp_min.set_index(pd.to_datetime(df_temp_min['datetime']))
#df_temp_min.to_sql('data_min_temp', engine, if_exists='replace')
#
df_solar = pd.read_csv('./data/IDCJAC0016_086338_1800_Data.csv')
df_solar.columns = ['product_code', 'station_Number', 'year', 'month', 'day', 'daily_solar_exposure_MJ']
df_solar['datetime']=pd.to_datetime(df_solar[['year','month', 'day']])
df_solar = df_solar.set_index(pd.to_datetime(df_solar['datetime']))
#df_solar.to_sql('data_solar', engine, if_exists='replace')



df_weather = pd.concat([df_temp_min['minimum_temperature_C'],df_temp_max['maximum_temperature_C'],df_solar['daily_solar_exposure_MJ'], df_rain['rainfall_mm']], axis=1, join='inner')
df_weather.to_sql('data_weather', engine, if_exists='replace')


######






###### EDA #######

def identify_useable_stations(ped_stations):
    # Active stations, installed before Jan 01, 2018
    ped_stations_to_use = ped_stations[(ped_stations.installation_date<dt.datetime(2018,1,1,0,0,0)) & (ped_stations['status']=='A')].index
    return ped_stations_to_use

index = identify_useable_stations(ped_stations)

ped_historic.groupby(by='sensor_id')['date_time'].agg(["min","max"])



ped_stations
