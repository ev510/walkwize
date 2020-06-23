# %% codecell
import numpy as np
import pandas as pd
import osmnx as ox

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import datetime as dt
import pickle

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2

from patsy import dmatrices
import statsmodels.api as sm
import statsmodels.formula.api as smf

import scipy
import scipy.interpolate as interpolate
import statsmodels.api as sm
import statsmodels.formula.api as smf

import pydeck as pdk

# pushing to S3
import boto3
# pulling from S3 (idealy they two would use same library)
import s3fs







import os



################################################################################
def set_up_database():
    # .env file
    dbname='melbourne_db'
    username='emilyvoytek'
    pswd='founded1835'


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
    # Round to the next hour
    now -= dt.timedelta(hours = -10-1, minutes = now.minute, seconds = now.second, microseconds = now.microsecond)
    # Create next 24 hours
    future = pd.date_range(now, now+dt.timedelta(hours=24),freq='h').to_frame()
    # Prep for model input
    future = expand_time_index(future)
    future['hourly_counts']=0
    expr = "hourly_counts ~ month + day_of_week + sin_hour + cos_hour"
    y_future, X_future = dmatrices(expr, future, return_type='dataframe')
    return y_future, X_future

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
    ped_stations_to_use = ped_stations[
        (ped_stations.installation_date<dt.datetime(2018,1,1,0,0,0))
        & (ped_stations['status']=='A')
        & (ped_stations['latitude'] > -37.825)
        & (ped_stations['latitude'] < -37.805)
        & (ped_stations['longitude'] > 144.95)
        & (ped_stations['longitude'] < 144.97)].index
    return ped_stations_to_use

    len(ped_stations_to_use)

def scaled_poisson_regression(station_ID, sf_weather):
    ##---------------- Query SQL database --------------------##

    sql_query = """SELECT * FROM data_ped_historic WHERE sensor_id = {0} AND year < 2020 AND year > 2013;""".format(str(station_ID))
    ## Query inactive station
    df = pd.read_sql_query(sql_query,engine)
    ## TODO: This SHOULD be handled in the SQL creation now, but needs to be confirmed
    df['hourly_counts'] = df['hourly_counts'].replace(',','', regex=True).astype(float)
    df = df.set_index(pd.to_datetime(df['date_time']))

    df = pd.concat([df,df_weather],axis=1,join='inner').sort_index()

    ##---------------- Select features --------------------##
    # TODO: Should call expand_time_index
    ds = df.index.to_series()
    df_prepped = pd.DataFrame()
    df_prepped['month'] = ds.dt.month
    df_prepped['day_of_week'] = ds.dt.dayofweek

    df_prepped['sin_hour'] = np.sin(2*np.pi*ds.dt.hour/24)
    df_prepped['cos_hour'] = np.cos(2*np.pi*ds.dt.hour/24)

    df_prepped['hourly_counts']=df['hourly_counts']
    df_prepped = df_prepped.dropna()

    expr = "hourly_counts ~ month + day_of_week + sin_hour + cos_hour"

    #df['earlier']=df.index-dt.timedelta(hours=1)
    #df = df.drop(['date_time','index','id'], axis=1)
    #df['earlier_hourly_counts']=df['hourly_counts'].shift(periods=1, fill_value=0)




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
    poisson_predictions = poisson_training_results.get_prediction(X_test).summary_frame()['mean']

    poisson_training_results.rmse =  sm.tools.eval_measures.rmse(df_test['hourly_counts'], poisson_predictions)

    print(station_ID)

    return df_test, df_train, poisson_training_results, nb2_training_results, y_train, y_test, X_train, X_test

def create_comparison_dict():
    sql_query = """SELECT * FROM data_ped_historic WHERE year < 2020 AND year > 2014;"""
    ## Query inactive station

    df = pd.read_sql_query(sql_query,engine)
    compare = df.groupby(['sensor_id','month', 'day','time'])['hourly_counts'].agg(['mean'])
    compare_dict = compare.to_dict()
    return compare_dict
    #compare['mean']['April', 'Friday', 0]

def get_map_data():
    """
    Example: G, gdf_nodes, gdf_edges = get_map_data()
    """

    G = ox.graph_from_bbox(-37.82,-37.805,144.95, 144.97, network_type='walk')
    #fig, ax = ox.plot_graph(G, node_zorder=2, node_color='w', bgcolor='k')


    gdf_nodes, gdf_edges = ox.utils_graph.graph_to_gdfs(G, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True)
    gdf_edges['centroid_x'] = gdf_edges.apply(lambda r: r.geometry.centroid.x, axis=1)
    gdf_edges['centroid_y'] = gdf_edges.apply(lambda r: r.geometry.centroid.y, axis=1)
    return G, gdf_nodes, gdf_edges

def single_station_negative_binomial_regression(station_ID, sf_weather):
    ##---------------- Query SQL database --------------------##

    sql_query = """SELECT * FROM data_ped_historic WHERE sensor_id = {0} AND year < 2020 AND year > 2013;""".format(str(station_ID))
    ## Query inactive station
    df = pd.read_sql_query(sql_query,engine)
    ## TODO: This SHOULD be handled in the SQL creation now, but needs to be confirmed
    df['hourly_counts'] = df['hourly_counts'].replace(',','', regex=True).astype(float)
    df = df.set_index(pd.to_datetime(df['date_time']))

    df = pd.concat([df,df_weather],axis=1,join='inner').sort_index()

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
    pickle_byte_obj = pickle.dumps(obj)
    s3_resource.Object(bucket,key).put(Body=pickle_byte_obj)

def pickle_from_S3(key):
    # key = 'poisson.p'
    bucket='walkwize'
    data_location = 's3://{}/{}'.format(bucket, key)
    return pd.read_pickle(data_location)

################################################################################
################################################################################
################################################################################
##### WHERE STUFF RUNS ####
np.random.RandomState(42);

# Set up SQL database
engine = set_up_database()

# Get map data, dimmensions are hardwired
G, gdf_nodes, gdf_edges = get_map_data()
# Push to S3 bucket
pickle_to_S3('G.p',G)
pickle_to_S3('gdf_nodes.p',gdf_nodes)
pickle_to_S3('gdf_edges.p',gdf_edges)
# Pull from S3 bucket
G = pickle_from_S3('G.p')
nodes = pickle_from_S3('gdf_nodes.p')
edges = pickle_from_S3('gdf_edges.p')


df_weather = sql_query_weather()
df_weather.resample('1H').pad();

ped_stations = make_table_ped_station();
index = identify_useable_stations(ped_stations)
ped_station = ped_stations.loc[index]
pickle_to_S3('ped_stations.p',ped_station)


ps = pickle_from_S3('ped_stations.p')


index
# SMALL SUBSET
station_IDs= range(20,23)
# FULL SET
#TODO:UPDATE!
station_IDs = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 17, 18, 19, 20, 21,
             22, 23, 24, 26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 40, 41, 42, 43,
             44, 45, 46, 47, 48, 49, 50, 51, 52, 53]

# For now, I will save all model results.
trained_models = {}
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

compare_dict = create_comparison_dict()
mean_values_from_historic = pd.DataFrame.from_dict({SID: compare['mean'][SID,'June', 'Friday', 1] for SID in station_IDs}, orient='index')


ped_current = pd.read_json("https://data.melbourne.vic.gov.au/resource/d6mv-s43h.json")
ped_current = ped_current.groupby('sensor_id')['total_of_directions'].sum().to_frame()



ped_current = pd.concat([mean_values_from_historic,ped_current],join='inner',axis=1)
ped_current = ped_current.rename(columns={0:'mean','total_of_directions':'current'})
ped_current['scale']=ped_current['current']/ped_current['mean']
ped_current[ped_current['scale']>1]=1

ped_current

compare['mean'][22,'April', 'Friday', hour]

for SID in station_IDs:
    for hour in range(0,24):
        print(compare_dict['mean'][SID,'April', 'Friday', hour])




stations_summary
stations_summary['poisson_rmse'].mean()
stations_summary['poisson_rmse']/stations_summary['mean']






print(plt.style.available)
plt.style.use('dark_background')
plt.rcParams.update({'font.size': 22})



SID = 8
y_future, X_future = make_future()
poisson_predictions = poisson_training_results[SID].get_prediction(X_test[SID]).summary_frame()['mean']
nb2_predictions = nb2_training_results[SID].get_prediction(X_test[SID]).summary_frame()['mean']

# Sample time-series data. Consider highlighting zero-inflatedness.
a = plt.figure(figsize=(4,4));
axes = a.add_axes([.2, .2, .7, .7]);
axes.plot(y_train[SID],'.',label='training data');
#axes.plot(pd.to_datetime(train_datetime),predictions,'.')
axes.plot(y_test[SID],'.',label='testing data');
axes.plot(poisson_predictions,'.',label='poisson prediction');
#axes.plot(nb2_predictions,'.',label='nb2');
axes.m(737014, 737018);
axes.legend(fancybox = True, framealpha=0);
myFmt = mdates.DateFormatter('%Y-%m-%d')
axes.xaxis.set_major_formatter(myFmt)
axes.xaxis.set_major_locator(mdates.DayLocator(interval=1))   #to get a tick every 15 minutes
a.autofmt_xdate()
a.savefig('train.png',transparent =True, dpi=600)
a


# Sample prediction data.
b = plt.figure(figsize=(4,4));
axes = b.add_axes([.2, .2, .7, .7]);
future_predicted = poisson_training_results[SID].get_prediction(X_future).summary_frame()['mean']
axes.plot(future_predicted, label='forecast');
import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%H:00')
axes.xaxis.set_major_formatter(myFmt)
axes.legend(fancybox = True, framealpha=0);
b.autofmt_xdate()
b.savefig('predict.png', transparent = True,dpi=600)
b




####

sql_query = """SELECT * FROM data_ped_historic WHERE sensor_id = 1 AND year > 2019;"""
data = pd.read_sql_query(sql_query,engine)
data = data.set_index(pd.to_datetime(data['date_time']))
data



c = plt.figure(figsize=(12,4));
axes = c.add_axes([.2, .2, .7, .7]);
axes.plot(data['hourly_counts'], '.',label='forecast');
import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%H:00')
axes.xaxis.set_major_formatter(myFmt)
axes.legend(fancybox = True, framealpha=0);
c.autofmt_xdate()
axes.set_xlim(737450+112, 737457+112);

#c.savefig('predict.png', transparent = True,dpi=600)
axes.get_xlim()
c


## FIGURE D

#plt.rc('axes',edgecolor='w')

sql_query = """SELECT hourly_counts FROM data_ped_historic;"""
data = pd.read_sql_query(sql_query,engine)

# An "interface" to matplotlib.axes.Axes.hist() method
d = plt.figure(figsize=(12,4));
axes = d.add_axes([.3, .2, .6, .7]);
n, bins, patches = axes.hist(x=data['hourly_counts'], bins=180, rwidth=0.85, lw=0);
axes.set_title('All stations');
axes.set_xlabel('pedestrians per hour')
axes.set_ylabel('count')
axes.set_yticklabels(['0','0.5', '1 mil'])
axes.set_xlim(0, 2500);

d


d.savefig('histogram.png', transparent = True,dpi=800)






## HISTOGRAM
plt.style.use('dark_background')
plt.rcParams.update({'font.size': 17})

sql_query = """SELECT * FROM data_ped_historic WHERE sensor_id = 6 AND year >2013;"""
data = pd.read_sql_query(sql_query,engine)
SID22_data = data.set_index(pd.to_datetime(data['date_time']))

fig1 = plt.figure(figsize=(8,8),dpi=800);

ax[0] = fig1.add_axes([0.2, 0.85, 0.8, 0.15])
ax[1] = fig1.add_axes([0.2, 0.60, 0.8, 0.15])
ax[2] = fig1.add_axes([0.2, 0.35, 0.8, 0.15])
ax[3] = fig1.add_axes([0.2, 0.10, 0.8, 0.15])

# ax[0].boxplot(SID22_data['hourly_counts'].groupby(SID22_data.index.year).apply(list));
# ax[1].boxplot(SID22_data['hourly_counts'].groupby(SID22_data.index.month).apply(list));
# ax[2].boxplot(SID22_data['hourly_counts'].groupby(SID22_data.index.dayofweek).apply(list));
# ax[3].boxplot(SID22_data['hourly_counts'].groupby(SID22_data.index.hour).apply(list));

ax[0].plot(SID22_data['hourly_counts'].groupby(SID22_data.index.year).mean(),linewidth=3);
ax[1].plot(SID22_data['hourly_counts'].groupby(SID22_data.index.month).mean(),linewidth=3);
ax[2].plot(SID22_data['hourly_counts'].groupby(SID22_data.index.dayofweek).mean(),linewidth=3);
ax[3].plot(SID22_data['hourly_counts'].groupby(SID22_data.index.hour).mean(),linewidth=3);

# rotation = 0;
# ax[0].set_xticklabels(['2014','2015','2016','2017','2018','2019','2020'],rotation=rotation);
ax[1].set_xticks(range(1,13));
ax[1].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],rotation=rotation);
ax[2].set_xticklabels(['','Mon','Tue','Wed','Thu','Fri','Sat','Sun'],rotation=rotation);
# ax[3].set_xticklabels(range(0,23),rotation=rotation);
ax[3].set_xticks(range(0,24,2));

ax[0].set_xlabel('year');
ax[1].set_xlabel('month');
ax[2].set_xlabel('day of week');
ax[3].set_xlabel('hour of day');

ax[0].set_ylim((800,1700))
ax[1].set_ylim((800,1700))
ax[2].set_ylim((800,1700))
ax[3].set_ylim((-500,5500))



for i in range(0,4):
    ax[i].set_ylabel('avg.\nped/hr',labelpad=30,rotation=0)
    ax[i].set_title('')

fig1
fig1.savefig('Station06.png',transparent = True,dpi=800)




for i in range(0,4):
    ax[i].set_ylim(0,2000)
    ax[i].set_ylabel('peds\nper\nhour',rotation = 0,labelpad=50)
ax[3].set_ylim(0,4000)

ax[0].set_xlabel('year')
ax[1].set_xlabel('month')
ax[2].set_xlabel('day of week')
ax[3].set_xlabel('hour of day')

fig
fig.savefig('overall_trends.png', transparent = True,dpi=400)


plt.plot(SID22_data.groupby('month')['hourly_counts'].mean().to_frame())


boxplot = df.boxplot(by='X')








SID = 22


## PLOT OF COVID TREND
plt.style.use('dark_background')
plt.rcParams.update({'font.size': 20})
sql_query = """SELECT * FROM data_ped_historic WHERE sensor_id = 22 AND year > 2019;"""
data = pd.read_sql_query(sql_query,engine)
data = data.set_index(pd.to_datetime(data['date_time']))


# Sample time-series data. Consider highlighting zero-inflatedness.
e = plt.figure(figsize=(12,4));
axes = e.add_axes([.2, .3, .7, .6]);
axes.plot(data['hourly_counts'],'.');
myFmt = mdates.DateFormatter('%b %Y')
axes.xaxis.set_major_formatter(myFmt)
e.autofmt_xdate()
axes.set_ylabel('hourly \ncount',rotation=0,labelpad=50)
axes.set_title('Station 22: Flinders St & Elizabeth St')
axes.get_xlim()
axes.set_ylim((-250,6000))

#
ped_current = pd.read_json("https://data.melbourne.vic.gov.au/resource/d6mv-s43h.json")
ped_current = ped_current.groupby('sensor_id')['total_of_directions'].sum().to_frame()
ped_current = ped_current.join(ped_stations[['latitude','longitude']])
#
axes.set_xlim(dt.datetime(year=2020,month=1,day=1),dt.datetime(year=2020,month=6,day=30))
now = dt.datetime.utcnow()- dt.timedelta(hours = -10)
#axes.plot(now,ped_current[ped_current.index==22]['total_of_directions'],'.');
axes.legend(fancybox = True, framealpha=0);

e.savefig('2020_trend.png', transparent = True,dpi=600)
e




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
