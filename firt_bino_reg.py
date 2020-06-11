
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2

import pandas as pd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

import datetime as dt
import pickle

################################################################################
def set_up_database():
    dbname = 'melbourne_db'
    username = 'emilyvoytek'
    pswd = 'founded1835'

    ## 'engine' is a connection to a database
    engine = create_engine('postgresql://%s:%s@localhost/%s'%(username,pswd,dbname))
    # Replace localhost with IP address if accessing a remote server\
    return engine

################################################################################
def make_future():
    # Creating the next 24 hours of AUSTRALIA time
    now = dt.datetime.utcnow()
    # Round to the next hour
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

################################################################################
def single_station_negative_binomial_regression(station_ID):

    ##---------------- Query SQL database --------------------##
    sql_query = """SELECT * FROM data_ped_historic WHERE sensor_id = {0} AND year < 2020 AND year > 2016;""".format(str(station_ID))
    ## Query inactive station
    df = pd.read_sql_query(sql_query,engine)
    ## TODO: This SHOULD be handled in the SQL creation now, but needs to be confirmed
    df['hourly_counts'] = df['hourly_counts'].replace(',','', regex=True).astype(float)
    df = df.set_index(pd.to_datetime(df['date_time']))


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
    ## features for model:
    expr = "hourly_counts ~ month + day_of_week + sin_hour + cos_hour"



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

################################################################################
## TODO: CURRENTLY BROKEN ##
def plotting_nbr_results(poisson_training_results,nb2_training_results):
    poisson_predictions = poisson_training_results.get_prediction(X_test).summary_frame()['mean']
    nb2_predictions = nb2_training_results.get_prediction(X_test).summary_frame()['mean']
    a = plt.figure();
    axes = a.add_axes([.1, .1, .8, .8]);
    axes.plot(df_train['hourly_counts'],'.',label='data_train');
    #axes.plot(pd.to_datetime(train_datetime),predictions,'.')
    axes.plot(df_test['hourly_counts'],'.',label='data_test');
    axes.plot(poisson_predictions,'.',label='poisson');
    axes.plot(nb2_predictions,'.',label='nb2');
    axes.set_xlim(737017, 737021);
    axes.legend();
    return a;



################################################################################
################################################################################
################################################################################
##### WHERE STUFF RUNS ####

engine = set_up_database()


# For now, I will save all model results.
poisson_training_results = {}
nb2_training_results = {}
df_test = {}
df_train = {}
y_train = {}
y_test = {}
X_train = {}
X_test = {}

#station_IDs= range(20,30)
station_IDs = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 17, 18, 19, 20, 21,
            22, 23, 24, 26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 40, 41, 42, 43,
            44, 45, 46, 47, 48, 49, 50, 51, 52, 53]

for SID in station_IDs:
    df_test[SID], df_train[SID], poisson_training_results[SID], nb2_training_results[SID],y_train[SID], y_test[SID], X_train[SID], X_test[SID] = single_station_negative_binomial_regression(SID)

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


stations_summary['poisson_rmse']/stations_summary['mean']


future_predicted = poisson_training_results[22].get_prediction(X_future).summary_frame()['mean']



a = plt.figure();
axes = a.add_axes([.1, .1, .8, .8]);
axes.plot(poisson_training_results[22].get_prediction(X_future).summary_frame()['mean'],'.',label='f');
axes.legend();
a




pickle.dump( [df_test, df_train, poisson_training_results, nb2_training_results,y_train,y_test,X_train,X_test], open( "save.p", "wb" ) )
#pickle.dump( [poisson_training_results], open( "poisson_model_results.p", "wb" ) )


[df_test, df_train, poisson_training_results, nb2_training_results,y_train,y_test,X_train,X_test] = pickle.load( open( "save.p", "rb" ) )



df_rain = pd.read_csv('./data/IDCJAC0009_086338_1800_Data.csv')
df_rain.columns = ['product_code', 'station_number', 'year', 'month', 'day', 'rainfall_mm','days_measured', 'quality']
df_rain['datetime']=pd.to_datetime(df_rain[['year','month', 'day']])
df_rain = df_rain.set_index(pd.to_datetime(df_rain['datetime']))

#
df_temp_max = pd.read_csv('./data/IDCJAC0010_086338_1800_Data.csv')
df_temp_max.columns = ['product_code', 'station_number', 'year', 'month', 'day', 'maximum_temperature_C',
       'days_measured', 'quality']
df_temp_max['datetime']=pd.to_datetime(df_temp_max[['year','month', 'day']])
df_temp_max = df_temp_max.set_index(pd.to_datetime(df_temp_max['datetime']))

#
df_temp_min = pd.read_csv('./data/IDCJAC0011_086338_1800_Data.csv')
df_temp_min.columns = ['product_code', 'station_Number', 'year', 'month', 'day', 'minimum_temperature_C',
       'days_measured', 'quality']
df_temp_min['datetime']=pd.to_datetime(df_temp_min[['year','month', 'day']])
df_temp_min = df_temp_min.set_index(pd.to_datetime(df_temp_min['datetime']))


#
df_solar = pd.read_csv('./data/IDCJAC0016_086338_1800_Data.csv')
df_solar.columns = ['product_code', 'station_Number', 'year', 'month', 'day', 'daily_solar_exposure_MJ']
df_solar['datetime']=pd.to_datetime(df_solar[['year','month', 'day']])
df_solar = df_solar.set_index(pd.to_datetime(df_solar['datetime']))

df_solar[]


#
#
# weather = .to_frame()
# weather.columns = ['date_time']
# weather = weather.set_index(pd.to_datetime(weather['date_time']))
# weather = weather.drop('date_time', axis=1)


weather = pd.concat([df_temp_min['minimum_temperature_C'],df_temp_max['maximum_temperature_C'],df_solar['daily_solar_exposure_MJ'], df_rain['rainfall_mm']], axis=1, join='inner')
plt.plot(weather)



a = plotting_nbr_results(poisson_training_results[22],nb2_training_results[22])

weather.resample('1H').pad()
df_temp_min['minimum_temperature_C'].resample('1H').bfill()




np.random.RandomState(42)


    sql_query = """SELECT MIN() FROM data_ped_historic WHERE sensor_id = {0} AND year < 2020 AND year > 2016;""".format(str(station_ID))

    sql_query2 = """SELECT sensor_id, MIN(date_time), MAX(date_time) FROM data_ped_historic GROUP BY sensor_id"""
    df = pd.read_sql_query(sql_query2,engine)
