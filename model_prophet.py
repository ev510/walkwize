from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import fbprophet as Prophet


plt.rcParams['font.size']=14


dbname = 'melbourne_db'
username = 'emilyvoytek'
pswd = 'founded1835'


## 'engine' is a connection to a database
## Here, we're using postgres, but sqlalchemy can connect to other things too.
engine = create_engine('postgresql://%s:%s@localhost/%s'%(username,pswd,dbname));
print('postgresql://%s:%s@localhost/%s'%(username,pswd,dbname));
# Replace localhost with IP address if accessing a remote server


# Sample SQL query
sql_query = """SELECT * FROM data_ped_stations WHERE sensor_id = '1';"""
df = pd.read_sql_query(sql_query,engine)


## HOLIDAYS ##
# from datetime import date
# import holidays
#
# vic_holidays = holidays.Australia(prov='Vic')
# vic = vic_holidays['2010-01-01':'2022-01-01']
# holidays = pd.DataFrame({
#   'holiday': 'Victoria',
#   'ds': pd.to_datetime([obj.strftime('%Y-%m-%d') for obj in vic])})


def run_FB_prophet(SID):
    #sql_query = """SELECT date_time, hourly_counts FROM data_ped_historic WHERE sensor_id = '13' AND year <'2020' AND year >'1990';"""
    print("Station {}: Query data".format(SID))
    sql_query = """SELECT date_time, hourly_counts FROM data_ped_historic WHERE sensor_id = '{}';""".format(SID)
    df = pd.read_sql_query(sql_query,engine)
    df.columns= ['ds','y']
    df['y'] = df['y'].replace(',','', regex=True).astype(float)
    print("Station {}: Initial Prophet model".format(SID))

    ### Running Prophet
    m_default = Prophet.Prophet()
    m_default.fit(df)
    #future_default = m.make_future_dataframe(freq='H',periods=2400)
    #forecast_default = m.predict(future)

    cps_orig = pd.DataFrame(m_default.changepoints)
    cps = cps_orig.append({'ds':'2020-01-01 00:00:00'}, ignore_index=True);
    cps = cps.append({'ds':'2020-02-01 00:00:00'}, ignore_index=True);
    cps = cps.append({'ds':'2020-03-01 00:00:00'}, ignore_index=True);
    cps = cps.append({'ds':'2020-04-01 00:00:00'}, ignore_index=True);
    cps = cps.append({'ds':'2020-04-15 00:00:00'}, ignore_index=True);
    cps = cps.append({'ds':'2020-05-01 00:00:00'}, ignore_index=True);
    cps = cps.append({'ds':'2020-05-11 00:00:00'}, ignore_index=True);
    cps = cps.append({'ds':'2020-05-21 00:00:00'}, ignore_index=True);

    print("Station {}: Modified Prophet model".format(SID))
    # second prophet model with added change points
    m = Prophet.Prophet(seasonality_mode='multiplicative',changepoints=cps['ds'])
    m.fit(df)

    future = m.make_future_dataframe(freq='H',periods=2400)
    forecast = m.predict(future)

    return m, future, forecast




    fig1 = m.plot_components(forecast);
    #fig2 = m.plot(forecast)
def plot_prohet(SID,future,forecast):    ## Prophet plot
    figsize=(10,6)
    xlabel='ds'
    ylabel='y'
    plt.style.use('default')

    fig3 = plt.figure(facecolor='w',figsize=figsize)

    #ax = fig.add_suplot(111)
    ax = fig.add_axes([.2, .3, .7, .6]);
    ax.plot(m.history['ds'].dt.to_pydatetime(), m.history['y'], 'k.')

    #ax.plot(df['ds'],df['y'],'.')
    ax.plot(future, forecast['yhat'], ls='-', c='#0072B2')
    # if 'cap' in fcst and plot_cap:
        #     ax.plot(future, fcst['cap'], ls='--', c='k')
        # if m.logistic_floor and 'floor' in fcst and plot_cap:
        #     ax.plot(fcst_t, fcst['floor'], ls='--', c='k')
        # if uncertainty:
        #     ax.fill_between(fcst_t, fcst['yhat_lower'], fcst['yhat_upper'],
                            # color='#0072B2', alpha=0.2)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig3.tight_layout()
    ax.get_xlim()

    ax.set_xlim(737460,737590-100)
    ax.set_xlim(737460,737580)

    return fig_forecast
def plot_prophet_components(SID,forecast):
    ### Modify Prophet component plot
    plt.style.use('dark_background')
    plt.style.use('default')

    plt.rcParams.update({'font.size': 10})

    plt.style.use('default')

    fig = m.plot_components(forecast,figsize=(4,6));

    for i in range(0,4):
        fig.axes[i].set_ylim(-1600,1600)

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

for SID in range(0,1):
    run_FB_prophet(SID)


fig.savefig('prophet_reformat.png',transparent=True,dpi=400)
