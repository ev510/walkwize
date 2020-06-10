
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
    # Creating the next 24 hours
    now = dt.datetime.now()
    # Round to the next hour
    now -= datetime.timedelta(hours = -1, minutes = now.minute, seconds = now.second, microseconds = now.microsecond)
    # Create next 24 hours
    future_times = pd.date_range(now, end=dt.datetime.now()+dt.timedelta(hours=24),freq='h').to_frame()
    return future_times

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


    nb2_predictions = nb2_training_results.get_prediction(X_test)
    predictions_summary_frame = nb2_predictions.summary_frame()
    #print(predictions_summary_frame)

    predicted_counts=predictions_summary_frame['mean']
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


set_up_database;
# For now, I will save all model results.
poisson_training_results = {}
nb2_training_results = {}
df_test = {}
df_train = {}
y_train = {}
y_test = {}
X_train = {}
X_test = {}

station_IDs= range(24,27)
for SID in station_IDs:
    df_test[SID], df_train[SID], poisson_training_results[SID], nb2_training_results[SID],y_train[SID], y_test[SID], X_train[SID], X_test[SID] = single_station_negative_binomial_regression(SID)

stations_summary = pd.DataFrame(columns=['poisson_rmse','nb2_rmse','training_length','min','max','mean' ])
for SID in station_IDs:
    row = pd.Series({'poisson_rmse':poisson_training_results[SID].rmse,
                        'nb2_rmse':nb2_training_results[SID].rmse,
                        'training_length':len(df_train[SID]),
                        'min':df_train[SID]['hourly_counts'].min(),
                        'max':df_train[SID]['hourly_counts'].max(),
                        'mean':df_train[SID]['hourly_counts'].mean().round(),
                        },
                        name=SID)
    stations_summary = stations_summary.append(row)

df_train[22]['hourly_counts'].mean()


b = poisson_training_results.keys()
b
stations_summary




a = plotting_nbr_results(poisson_training_results[22],nb2_training_results[22])





np.random.RandomState(42)


    sql_query = """SELECT MIN() FROM data_ped_historic WHERE sensor_id = {0} AND year < 2020 AND year > 2016;""".format(str(station_ID))

    sql_query2 = """SELECT sensor_id, MIN(date_time), MAX(date_time) FROM data_ped_historic GROUP BY sensor_id"""
    df = pd.read_sql_query(sql_query2,engine)
