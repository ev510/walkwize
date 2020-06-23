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


##### MODELING ####
def expand_time_index(df):

    ds = df.index.to_series()
    df['month'] = ds.dt.month
    df['day_of_week'] = ds.dt.dayofweek
    #df_prepped['day']= ds.dt.day
    #df_prepped['hour']=ds.dt.hour
    df['sin_hour'] = np.sin(2*np.pi*ds.dt.hour/24)
    df['cos_hour'] = np.cos(2*np.pi*ds.dt.hour/24)
    return df
