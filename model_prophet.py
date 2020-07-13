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
engine = create_engine('postgresql://%s:%s@localhost/%s'%(username,pswd,dbname))
print('postgresql://%s:%s@localhost/%s'%(username,pswd,dbname))
# Replace localhost with IP address if accessing a remote server


# Sample SQL query
sql_query = """SELECT * FROM data_ped_stations WHERE sensor_id < '10';"""

df = pd.read_sql_query(sql_query,engine)




### Prepping for Prophet
sql_query = """SELECT date_time, hourly_counts FROM data_ped_historic WHERE sensor_id = '43' AND year <'2020' AND year >'1990';"""
df = pd.read_sql_query(sql_query,engine)
df.date_time.min()
df.columns= ['ds','y']
df['y'] = df['y'].replace(',','', regex=True).astype(float)

### Running Prophet
m = Prophet.Prophet()
m.fit(df)
future = m.make_future_dataframe(freq='H',periods=2400)
forecast = m.predict(future)

### Plotting Prophet
m.plot(forecast);
fig = m.plot_components(forecast);


fig.savefig('prophet2.png',transparent=True,dpi=400)




def default_Prophet_plot(arg):

    figsize=(10,6)
    xlabel='ds'
    ylabel='y'

    fig = plt.figure(facecolor='w',figsize=figsize)

    ax = fig.add_suplot(111)

    ax.plot(fcst_t, fcst['yhat'], ls='-', c='#0072B2')
    if 'cap' in fcst and plot_cap:
        ax.plot(fcst_t, fcst['cap'], ls='--', c='k')
    if m.logistic_floor and 'floor' in fcst and plot_cap:
        ax.plot(fcst_t, fcst['floor'], ls='--', c='k')
    if uncertainty:
        ax.fill_between(fcst_t, fcst['yhat_lower'], fcst['yhat_upper'],
                        color='#0072B2', alpha=0.2)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig










## This might be helpful in extracting frequency decomposition from Prophet model
days = (pd.date_range(start='2019-01-01', periods=365) + pd.Timedelta(days=0))
df_y = m.seasonality_plot_df(days)
seas = model.predict_seasonal_components(df_y)

## This one, too.
def my_custom_plot_weekly(m, ax=None, uncertainty=True, weekly_start=0, figsize=(10, 6), name='weekly'):
    """Plot the weekly component of the forecast.
    Parameters
    ----------
    m: Prophet model.
    ax: Optional matplotlib Axes to plot on. One will be created if this
        is not provided.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    weekly_start: Optional int specifying the start day of the weekly
        seasonality plot. 0 (default) starts the week on Sunday. 1 shifts
        by 1 day to Monday, and so on.
    figsize: Optional tuple width, height in inches.
    name: Name of seasonality component if changed from default 'weekly'.
    Returns
    Adapted from:
    -------
    """
    artists = []
    if not ax:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    # Compute weekly seasonality for a Sun-Sat sequence of dates.
    days = (pd.date_range(start='2017-01-01', periods=7) +
            pd.Timedelta(days=weekly_start))
    # Import this function: seasonality_plot_df
    df_w = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_w)
    days = days.weekday_name
    # Return the data here, do not plot.
    return days, seas
