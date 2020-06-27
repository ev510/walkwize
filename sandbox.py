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


##### MODELING ####

def expand_time_index(df):
    ds = df.index.to_series()
    df['month'] = ds.dt.month
    df['day_of_week'] = ds.dt.dayofweek
    #df_prepped['day']= ds.dt.day
    # df_prepped['hour']=ds.dt.hour
    df['sin_hour'] = np.sin(2 * np.pi * ds.dt.hour / 24)
    df['cos_hour'] = np.cos(2 * np.pi * ds.dt.hour / 24)
    return df



@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_model_results_data():
    bucket = 'walkwize'
    data_key = 'poisson.p'
    data_location = 's3://{}/{}'.format(bucket, data_key)
    #poisson_training_results = pd.read_pickle(data_location)
    return pd.read_pickle(data_location)

# def get_ped_predicted(model_training_results):
#     #[df_test, df_train, poisson_training_results, nb2_training_results,y_train,y_test,X_train,X_test] = pickle.load( open( "poisson.p", "rb" ) )#
#
#     station_IDs = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 17, 18, 19, 20, 21,
#                    22, 23, 24, 26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 40, 41, 42, 43,
#                    44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
#     #
#     y_future, X_future = make_future()
#
#     for SID in station_IDs:
#         y_future.insert(1, SID, model_training_results[SID].get_prediction(
#             X_future).summary_frame()['mean'], True)
#
#     y_future = y_future.drop('hourly_counts', axis=1)
#     ped_predicted = y_future.transpose()
#     type(ped_predicted.index)
#     return ped_predicted
#     # ped_predicted.index.is_numeric()
#


#



# def make_future():
#     # Creating the next 24 hours of AUSTRALIA time
#     now = dt.datetime.utcnow()
#     # Round to the next hour
#     now -= dt.timedelta(hours=-10 - 1, minutes=now.minute,
#                         seconds=now.second, microseconds=now.microsecond)
#     # Create next 24 hours
#     future = pd.date_range(
#         now, now + dt.timedelta(hours=24), freq='h').to_frame()
#     # Prep for model input
#     future = expand_time_index(future)
#     future['hourly_counts'] = 0
#     expr = "hourly_counts ~ month + day_of_week + sin_hour + cos_hour"
#     y_future, X_future = dmatrices(expr, future, return_type='dataframe')
#     return y_future, X_future
#
#
