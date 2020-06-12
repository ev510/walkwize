#!/usr/bin/env python
# coding: utf-8
source activate insight

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import fbprophet as Prophet




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

pd.read_sql_query(sql_query,engine)


sql_query = """
SELECT date_time, hourly_counts FROM data_ped_historic WHERE sensor_id = '43' AND year <'2020' AND year >'1990';
"""

df = pd.read_sql_query(sql_query,engine)


df.date_time.min()


df.columns= ['ds','y']

df['y'] = df['y'].replace(',','', regex=True).astype(float)





import fbprophet
m = fbprophet.Prophet()




m.fit(df)



future = m.make_future_dataframe(freq='H',periods=100)
future


# In[12]:


forecast = m.predict(future)







fig = plt.figure(facecolor='w',figsize=figsize)





figsize=(10,6)
xlabel='ds'
ylabel='y'

fig = plt.figure(facecolor='w',figsize=figsize)

ax = fig.add_suplot(111)


# In[ ]:




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


# In[ ]:


m.plot_components(forecast)


# In[ ]:


m.plot_components(forecast)


# In[ ]:


# I save my iPython scripts as pure python scripts for versioning in git.
# They are much cleaner.
get_ipython().system('jupyter nbconvert --to=python Processing_timelapse.ipynb')


# In[ ]:







# SARIMAX example
from statsmodels.tsa.statespace.sarimax import SARIMAX
from random import random
# contrived dataset
data1 = [x + random() for x in range(1, 100)]
data2 = [x + random() for x in range(101, 200)]
# fit model
model = SARIMAX(data1, exog=data2, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
model_fit = model.fit(disp=False)
# make prediction
exog2 = [200 + random()]
yhat = model_fit.predict(len(data1), len(data1), exog=[exog2])
print(yhat)
