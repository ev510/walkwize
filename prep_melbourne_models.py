# %% codecell
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2

from patsy import dmatrices
import statsmodels.api as sm
import statsmodels.formula.api as smf

import datetime as dt
import pickle

#pushing to S3
import boto3
# pulling from S3 (idealy they two would use same library)
import s3fs





# %% codecell

################################################################################
def set_up_database():
    dbname = 'melbourne_db'
    username = 'emilyvoytek'
    pswd = 'founded1835'

    ## 'engine' is a connection to a database
    engine = create_engine('postgresql://%s:%s@localhost/%s'%(username,pswd,dbname))
    # Replace localhost with IP address if accessing a remote server
    return engine

###### LOAD HISTORIC WEATHER DATA ######
def make_table_weather():
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
    #
    # COMBINED WEATHER DB
    df_weather = pd.concat([df_temp_min['minimum_temperature_C'],df_temp_max['maximum_temperature_C'],df_solar['daily_solar_exposure_MJ'], df_rain['rainfall_mm']], axis=1, join='inner')
    df_weather = df_weather.resample('1H').pad()
    df_weather.to_sql('data_weather', engine, if_exists='replace')

def make_table_ped_historic():
    ######  LOAD THE HISTORIC PED DATA  ######
    #stations = pd.read_json("https://data.melbourne.vic.gov.au/resource/b2ak-trbp.json?$limit=10000000")
    #ped_historic = stations

    #len(stations)
    #stations.head()
    ped_historic = pd.read_csv('./data/Pedestrian_Counting_System___2009_to_Present__counts_per_hour_June04.csv')
    ped_historic.columns=(['id', 'date_time', 'year', 'month', 'mdate', 'day', 'time', 'sensor_id','sensor_name', 'hourly_counts'])
    ped_historic['hourly_counts'] = ped_historic['hourly_counts'].replace(',','', regex=True).astype(float)

    # ATTENTION: THIS TAKES FOREVER (~5-10 min) TO RUN
    # ped_historic.to_sql('data_ped_historic', engine, if_exists='replace')

def make_table_ped_station():
    ######  LOAD PED STATION INFORMATION  ######
    ped_stations = pd.read_csv('./data/Pedestrian_Counting_System_Sensor_Locations.csv')
    ped_stations['installation_date'] = pd.to_datetime(ped_stations['installation_date'])

    ped_stations.set_index("sensor_id",inplace=True)
    ped_stations = ped_stations.sort_index()
    ped_stations.to_sql('data_ped_stations', engine, if_exists='replace')
    return ped_stations

################################################################################
def sql_query_weather():
    ##---------------- Query SQL database --------------------##
    sql_query = """SELECT * FROM data_weather;"""
    ## Query inactive station
    df_weather = pd.read_sql_query(sql_query,engine)
    ## TODO: This SHOULD be handled in the SQL creation now, but needs to be confirmed
    df_weather = df_weather.set_index(df_weather['datetime'])
    return df_weather

def sql_query_ped_historic():
    sql_query = """SELECT MIN() FROM data_ped_historic WHERE sensor_id = {0} AND year < 2020 AND year > 2016;""".format(str(station_ID))
    sql_query2 = """SELECT sensor_id, MIN(date_time), MAX(date_time) FROM data_ped_historic GROUP BY sensor_id"""
    return pd.read_sql_query(sql_query2,engine)

################################################################################
def make_future():
    # Creating the next 24 hours of AUSTRALIA time
    now = dt.datetime.utcnow()
    # Round to the next hour (-1), shift to Australia time (-10)
    now -= dt.timedelta(hours = -11, minutes = now.minute, seconds = now.second, microseconds = now.microsecond)
    # Create next 24 hours
    future_times = pd.date_range(now, now+dt.timedelta(hours=24),freq='h')
    type(future_times)

    future = make_future().to_frame()
    future = expand_time_index(future)
    future['hourly_counts']=0
    expr = "hourly_counts ~ month + day_of_week + sin_hour + cos_hour"
    y_future, X_future = dmatrices(expr, future, return_type='dataframe')
    return X_future

################################################################################
def expand_time_index(df):
    ds = df.index.to_series()
    df['month'] = ds.dt.month
    df['day_of_week'] = ds.dt.dayofweek
    #df_prepped['day']= ds.dt.day
    #df_prepped['hour']=ds.dt.hour
    df['sin_hour'] = np.sin(2*np.pi*ds.dt.hour/24)
    df['cos_hour'] = np.cos(2*np.pi*ds.dt.hour/24)
    return df

###### Identify stations to model #######
def identify_useable_stations(ped_stations):
    # Active stations, installed before Jan 01, 2018
    ped_stations_to_use = ped_stations[(ped_stations.installation_date<dt.datetime(2018,1,1,0,0,0)) & (ped_stations['status']=='A')].index
    return ped_stations_to_use







def single_station_negative_binomial_regression(station_ID, sf_weather):
    ##---------------- Query SQL database --------------------##

    sql_query = """SELECT * FROM data_ped_historic WHERE sensor_id = {0} AND year < 2020 AND year > 2013;""".format(str(station_ID))
    ## Query inactive station
    df = pd.read_sql_query(sql_query,engine)
    ## TODO: This SHOULD be handled in the SQL creation now, but needs to be confirmed
    df['hourly_counts'] = df['hourly_counts'].replace(',','', regex=True).astype(float)
    df = df.set_index(pd.to_datetime(df['date_time']))

    df = pd.concat([df,df_weather],axis=1,join='inner').sort_index()
    df['earlier']=df.index-dt.timedelta(hours=1)

    df = df.drop(['date_time','index','id'], axis=1)
    df['earlier_hourly_counts']=df['hourly_counts'].shift(periods=1, fill_value=0)


    #
    # compare = df.groupby(['month', 'day','time'])['hourly_counts'].agg(['mean'])
    # compare
    # #compare.loc(compare['month']=='April')
    #
    #
    # df_gb = df.groupby(['Factor1','Factor2'])['Values'].aggregate([np.mean, np.std])
    # df_sum = df_gb.groupby(['Factor1','Factor2']).aggregate(np.sum)
    # df_sum.to_dict()
    # type(compare)
    # compare.items()
    #
    #
    # # Drop first time step, no historic data
    #
    #
    #

    ##---------------- Select features --------------------##
    # TODO: Should call expand_time_index
    ds = df.index.to_series()
    df_prepped = pd.DataFrame()
    df_prepped['month'] = ds.dt.month
    df_prepped['day_of_week'] = ds.dt.dayofweek
    #df_prepped['day']= ds.dt.day
    #df_prepped['hour']=ds.dt.hour
    df_prepped['sin_hour'] = np.sin(2*np.pi*ds.dt.hour/24)
    df_prepped['cos_hour'] = np.cos(2*np.pi*ds.dt.hour/24)
    #df_prepped['sin_day'] = np.sin(2*np.pi*ds.dt.dayofweek/7)
    #df_prepped['cos_day'] = np.cos(2*np.pi*ds.dt.dayofweek/7)
    #df_prepped['is_weekday'] = ((ds.dt.dayofweek) // 5 == 0).astype(float)
    df_prepped['hourly_counts']=df['hourly_counts']
    #df_prepped['minimum_temperature_C']=df['minimum_temperature_C']
    #df_prepped['maximum_temperature_C']=df['maximum_temperature_C']
    #df_prepped['daily_solar_exposure_MJ']=df['daily_solar_exposure_MJ']
    #df_prepped['rainfall_mm']=df['rainfall_mm']
    df_prepped = df_prepped.dropna()

    ## features for model:
    #expr = "hourly_counts ~ month + day_of_week + sin_hour + cos_hour + minimum_temperature_C + maximum_temperature_C + daily_solar_exposure_MJ + rainfall_mm"
    expr = "hourly_counts ~ month + day_of_week + sin_hour + cos_hour"

    df_prepped

    ##---------------- Split data set -------------------##
    mask = np.random.rand(len(df_prepped)) < 0.8
    df_train = df_prepped[mask]
    df_test = df_prepped[~mask]
    #print('Training data set length='+str(len(df_train)));
    #print('Testing data set length='+str(len(df_test)));
    y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
    y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')

    ##---------------- Build model --------------------##
    poisson_training_results = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()

    #len(poisson_training_results.mu)
    #len(df_train)


    df_train.loc[:,'hourly_lambda']=poisson_training_results.mu.copy();
    df_train.loc[:,'AUX_OLS_DEP']=df_train.apply(lambda x: ((x['hourly_counts'] - x['hourly_lambda'])**2 - x['hourly_counts']) / x['hourly_lambda'], axis=1);

    ols_expr = """AUX_OLS_DEP ~ hourly_lambda - 1"""
    aux_olsr_results = smf.ols(ols_expr, df_train).fit()
    #print(aux_olsr_results.params)
    #aux_olsr_results.tvalues

    nb2_training_results = sm.GLM(y_train, X_train,family=sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0])).fit();
    #print(nb2_training_results.summary())


    #nb2_predictions = nb2_training_results.get_prediction(X_test)
    #predictions_summary_frame = nb2_predictions.summary_frame()
    #print(predictions_summary_frame)

    #predicted_counts=predictions_summary_frame['mean']
    #actual_counts = y_test['hourly_counts']


    poisson_predictions = poisson_training_results.get_prediction(X_test).summary_frame()['mean']
    nb2_predictions = nb2_training_results.get_prediction(X_test).summary_frame()['mean']

    poisson_training_results.rmse =  sm.tools.eval_measures.rmse(df_test['hourly_counts'], poisson_predictions)
    nb2_training_results.rmse = sm.tools.eval_measures.rmse(df_test['hourly_counts'], nb2_predictions)

    print(station_ID)

    return df_test, df_train, poisson_training_results, nb2_training_results, y_train, y_test, X_train, X_test

def pickle_to_S3(key, obj):
    # key='poisson.p', obj=[poisson_training_results]
    s3_resource = boto3.resource('s3')
    bucket='walkwize'
    pickle_byte_obj = pickle.dumps([obj])
    s3_resource.Object(bucket,key).put(Body=pickle_byte_obj)

def pickle_from_S3(key):
    # key = 'poisson.p'
    bucket='walkwize'
    data_location = 's3://{}/{}'.format(bucket, key)
    return = pd.read_pickle(data_location)



################################################################################
################################################################################
################################################################################
##### WHERE STUFF RUNS ####
np.random.RandomState(42);
engine = set_up_database()

df_weather = sql_query_weather()
df_weather.resample('1H').pad();

ped_stations = make_table_ped_station();
index = identify_useable_stations(ped_stations)
# SMALL SUBSET
station_IDs= range(20,23)
# FULL SET
station_IDs = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 17, 18, 19, 20, 21,
             22, 23, 24, 26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 40, 41, 42, 43,
             44, 45, 46, 47, 48, 49, 50, 51, 52, 53]

# For now, I will save all model results.
poisson_training_results = {}
poisson_training_results_a = {}
poisson_training_results_b = {}

nb2_training_results = {}
df_test = {}
df_train = {}
y_train = {}
y_test = {}
X_train = {}
X_test = {}


for SID in station_IDs:
    df_test[SID], df_train[SID], poisson_training_results[SID], nb2_training_results[SID],y_train[SID], y_test[SID], X_train[SID], X_test[SID] = single_station_negative_binomial_regression(SID, df_weather)



### SUMMARY STATISTICS
stations_summary = pd.DataFrame(columns=['min','max','mean','training_length','poisson_rmse','nb2_rmse'])
for SID in station_IDs:
    row = pd.Series({'poisson_rmse':poisson_training_results[SID].rmse.round(),
                        'nb2_rmse':nb2_training_results[SID].rmse.round(),
                        'training_length':len(df_train[SID]),
                        'min':df_train[SID]['hourly_counts'].min(),
                        'max':df_train[SID]['hourly_counts'].max(),
                        'mean':df_train[SID]['hourly_counts'].mean().round(),
                        },
                        name=SID)
    stations_summary = stations_summary.append(row)






stations_summary
stations_summary['poisson_rmse'].mean()
stations_summary['poisson_rmse']/stations_summary['mean']


future_predicted = poisson_training_results[22].get_prediction(X_future).summary_frame()['mean']


# Save large pickle
pickle.dump( [df_test, df_train, poisson_training_results, nb2_training_results,y_train, y_test, X_train, X_test], open( "poisson2.p", "wb" ) )
# LOAD BIG PICKLE
[df_test, df_train, poisson_training_results, nb2_training_results,y_train,y_test,X_train,X_test] = pickle.load( open( "save.p", "rb" ) )

ped_historic.groupby(by='sensor_id')['date_time'].agg(["min","max"])
