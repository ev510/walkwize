from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import fbprophet as Prophet
import datetime as dt
import numpy as np

plt.rcParams['font.size']=14



# Replace localhost with IP address if accessing a remote server


# Sample SQL query

## HOLIDAYS ##
# from datetime import date
# import holidays
#
# vic_holidays = holidays.Australia(prov='Vic')
# vic = vic_holidays['2010-01-01':'2022-01-01']
# holidays = pd.DataFrame({
#   'holiday': 'Victoria',
#   'ds': pd.to_datetime([obj.strftime('%Y-%m-%d') for obj in vic])})


def query_SQL(SID):
    print("Station {}: Query data".format(SID))
    #sql_query = """SELECT date_time, hourly_counts FROM data_ped_historic WHERE sensor_id = '13' AND year <'2020' AND year >'1990';"""
    sql_query = """SELECT date_time, hourly_counts FROM data_ped_historic WHERE sensor_id = '{}' AND year > 2016;""".format(SID)
    df = pd.read_sql_query(sql_query,engine)
    df.columns= ['ds','y']
    df['y'] = df['y'].replace(',','', regex=True).astype(float)
    return df

def run_FB_prophet(df):
    # print("Station {}: Running baseline Prophet model".format(SID))
    # ### Running Prophet
    # m_baseline = Prophet.Prophet()
    # m_baseline.fit(df)
    # #future_default = m.make_future_dataframe(freq='H',periods=2400)
    # #forecast_default = m.predict(future)

    # cps_baseline = pd.DataFrame(m_baseline.changepoints)
    # cps = cps_baseline.append( [{'ds':'2020-01-01 00:00:00'},
    #                             {'ds':'2020-02-01 00:00:00'},
    #                             {'ds':'2020-02-15 00:00:00'},
    #                             {'ds':'2020-03-01 00:00:00'},
    #                             {'ds':'2020-03-11 00:00:00'},
    #                             {'ds':'2020-03-21 00:00:00'},
    #                             {'ds':'2020-04-01 00:00:00'},
    #                             {'ds':'2020-04-11 00:00:00'},
    #                             {'ds':'2020-04-21 00:00:00'}
    #                             #{'ds':'2020-05-01 00:00:00'},
    #                             #{'ds':'2020-05-11 00:00:00'},
    #                             #{'ds':'2020-05-21 00:00:00'}
    #                             ], ignore_index=True);

    print("Station {}: Running optimized Prophet model".format(SID))
    # second prophet model with added change points
    m_optimized = Prophet.Prophet(seasonality_mode='multiplicative',changepoint_range=1,changepoint_prior_scale = 1,n_changepoints=100)
    # changepoints=cps['ds'],
    m_optimized.fit(df)
    return m_optimized

def make_prophet_future():
    # Creating the next 24 hours of AUSTRALIA time
    now = dt.datetime.utcnow()
    # Round to the next hour
    now -= dt.timedelta(hours = -10-1, minutes = now.minute, seconds = now.second, microseconds = now.microsecond)
    # Create next 24 hours
    prophet_future = pd.date_range(now, now+dt.timedelta(hours=24),freq='h').to_frame()
    prophet_future = prophet_future.rename(columns={0:'ds'}).reset_index().drop('index',axis=1)



    # Prep for model input
    #future = expand_time_index(future)
    #future['hourly_counts']=0
    #expr = "hourly_counts ~ month + day_of_week + sin_hour + cos_hour"
    #y_future, X_future = dmatrices(expr, future, return_type='dataframe')
    return prophet_future

def predict_future(m, future):
    #future = m.make_future_dataframe(freq='H',periods=2400)
    forecast = m.predict(future)
    forecast['yhat'].clip_lower(0)
    return forecast

def plot_prohet(SID,m,forecast,test):    ## Prophet plot
    plt.style.use('dark_background')
    #plt.style.use('default')
    figsize=(12,4)
    plt.rcParams.update({'font.size': 20})
    fig = plt.figure(figsize=figsize);

    #ax = fig.add_suplot(111)
    ax = fig.add_axes([.2, .3, .7, .6]);
    ax.fill_between(forecast['ds'].dt.to_pydatetime(), forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2)
    ax.plot(m.history['ds'].dt.to_pydatetime(), m.history['y'], '.')
    ax.plot(pd.to_datetime(test['ds']), test['y'], '.')

    #ax.plot(forecast)
    #ax.plot(df['ds'],df['y'],'.')
    #ax.plot(future, forecast['yhat'], ls='-', c='#0072B2')
    # if 'cap' in fcst and plot_cap:
        #     ax.plot(future, fcst['cap'], ls='--', c='k')
        # if m.logistic_floor and 'floor' in fcst and plot_cap:
        #     ax.plot(fcst_t, fcst['floor'], ls='--', c='k')

    #ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_ylabel('avg.\nped/hr',labelpad=30,rotation=0)
    fig.tight_layout()
    ax.get_xlim()

    ax.set_xlim(737460,pd.to_datetime('2020-05-31'))
    #ax.set_xlim(737460,737580)
    #ax.set_xlim(737570,737620)
    fig_forecast = fig
    return fig_forecast

    fig_forecast


def plot_prophet_components(SID,m,forecast,test):
    ### Modify Prophet component plot
    plt.style.use('dark_background')
    plt.rcParams.update({'font.size': 10})

    fig = m.plot_components(forecast,figsize=(8,10));

    # for i in range(0,4):
    #     fig.axes[i].set_ylim(-1600,1600)

    fig.axes[0].set_xlabel('year');
    fig.axes[1].set_xlabel('day of week');
    fig.axes[2].set_xlabel('day of year');
    fig.axes[3].set_xlabel('time of day');

    fig.axes[0].set_ylabel('signal\nstrength', rotation=0);
    fig.axes[1].set_ylabel('signal\nstrength', rotation=0);
    fig.axes[2].set_ylabel('signal\nstrength', rotation=0);
    fig.axes[3].set_ylabel('signal\nstrength', rotation=0);

    fig.axes[0].set_xticklabels(['2009','','','2012','','','2015','','','2018'],rotation=0);
    fig.axes[1].set_xticklabels(['Sun','Mon','Tue','Wed','Thu','Fri','Sat'],rotation=0);
    fig.axes[2].set_xticklabels(['Jan','Mar','May','Jul','Sep','Nov','Jan'],rotation=0);
    fig.axes[3].set_xticklabels(['0:00','4:00','8:00','12:00','16:00','20:00','0:00'],rotation=0);

    fig_save0=fig.axes[0].get_position()
    fig_save1=fig.axes[1].get_position()
    fig_save2=fig.axes[2].get_position()
    fig_save3=fig.axes[3].get_position()

    fig.axes[1].set_position(fig_save2)
    fig.axes[2].set_position(fig_save1)

    fig.tight_layout()
    fig_components = fig
    return fig_components

############### RUN ALL THE THINGS ###############

dbname = 'melbourne_db'
username = 'emilyvoytek'
pswd = 'founded1835'


## 'engine' is a connection to a database
## Here, we're using postgres, but sqlalchemy can connect to other things too.
engine = create_engine('postgresql://%s:%s@localhost/%s'%(username,pswd,dbname));

SIDs = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 17, 18, 19, 20, 21,
            22, 23, 24, 26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 40, 41, 42, 43,
            44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
SIDs = [ 1,  2,  3,  4,  5,  6,  9, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 35, 36, 41, 45, 47, 48, 49, 51, 52, 53]
SIDs = [ 6]
SID = 4

for SID in SIDs:
    df = query_SQL(SID)


    # Sort (not strictly necessary)
    # df=df.set_index(pd.to_datetime(df['ds'])).sort_index()
    df['y_year_prior']=df['y'].shift(periods=52*7*24, fill_value=0)
    df['y_month_prior']=df['y'].shift(periods=28*24, fill_value=0)

    df_train = df[pd.to_datetime(df['ds'])<pd.to_datetime('2020-05-15')]
    df_test = df[pd.to_datetime(df['ds'])>=pd.to_datetime('2020-05-15')]

    m_optimized = run_FB_prophet(df_train);

    print("Station {}: Preparing forcasts".format(SID));

    df_predict = m_optimized.predict(df_test)
    df_predict['yhat'].clip(lower=0);



    df2_predict = df_predict.set_index(df_predict['ds']).sort_index()
    df2_predict = df2_predict.join(df.set_index(pd.to_datetime(df['ds'])).sort_index(),lsuffix='L')

    idx = (df2_predict.index>pd.to_datetime('2020-05-01')) & (df2_predict.index<pd.to_datetime('2020-06-01'))


    abs(df2_predict[idx]['y_month_prior']-df2_predict[idx]['y']).mean()

    abs(df2_predict[idx]['y_year_prior']-df2_predict[idx]['y']).mean()

    abs(df2_predict[idx]['y_hat']-df2_predict[idx]['y']).mean()





    idx = (pd.to_datetime(df_predict['ds'])>pd.to_datetime('2020-05-01')) & (pd.to_datetime(df_predict['ds'])<pd.to_datetime('2020-06-01'))
    (abs(df2_predict[idx]['y_month_prior']-df2_predict[idx]['y'])).mean()
    (abs(df[idx]['y_year_prior']-df[idx]['y'])).mean()
    (abs(df_predict['yhat']-df['y'])).mean()

    (abs(df[idx]['y_year_prior']-df[idx]['y'])).mean()
    (abs(df_predict['yhat']-df['y'])).mean()






    df2_predict
    df
    #m_optimized.plot(df_predict);

    print("Station {}: Generating figures".format(SID));
    fig = plot_prohet(SID,m_optimized,df_predict,df_test)
    fig.savefig('prophet_{}_changeprior1.png'.format(SID),transparent=True,dpi=400,)
    #m_optimized.plot_components(df_predict)
    fig = plot_prophet_components(SID,m_optimized,df_predict,df_test)
    fig.savefig('prophet_{}_components_changeprior1.png'.format(SID),transparent=True,dpi=400)
    print("Station {}: Done".format(SID))

    #plt.close('all')
