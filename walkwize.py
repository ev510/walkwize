import datetime as dt
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pickle
import pydeck as pdk
import scipy
import scipy.interpolate as interpolate
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st
import timeit
import s3fs
import boto3
import pytz

from patsy import dmatrices
## Credit to Dave Montiero of Doggo

def get_node_df(location):
    #Inputs: location as tuple of coords (lat, lon)
    #Returns: 1-line dataframe to display an icon at that location on a map

    #Location of Map Marker icon
    icon_data = {
        "url": "https://img.icons8.com/plasticine/100/000000/marker.png",
        "width": 128,
        "height":128,
        "anchorY": 128}

    return pd.DataFrame({'lat':[location[0]], 'lon':[location[1]], 'icon_data': [icon_data]})

def pickle_from_S3(key):
    # example: key = 'poisson.p'
    bucket='walkwize'
    data_location = 's3://{}/{}'.format(bucket, key)
    return pd.read_pickle(data_location)

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_ped_stations():
    # ped_stations = pd.read_json("https://data.melbourne.vic.gov.au/resource/h57g-5234.json")
    # ped_stations.set_index("sensor_id",inplace=True)
    return pickle_from_S3('ped_stations.p')

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_gdf_nodes():
    return pickle_from_S3('gdf_nodes.p')

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_gdf_edges():
    return pickle_from_S3('gdf_edges.p')

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_map_data():
    return pickle_from_S3('G.p')

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_trained_models():
    return pickle_from_S3('trained_models.p')

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_modeled_future():
    return pickle_from_S3('modeled_future.p')




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



def get_ped_predicted(model_training_results):
    #[df_test, df_train, poisson_training_results, nb2_training_results,y_train,y_test,X_train,X_test] = pickle.load( open( "poisson.p", "rb" ) )#

    station_IDs = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 17, 18, 19, 20, 21,
                22, 23, 24, 26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 40, 41, 42, 43,
                44, 45, 46, 47, 48, 49, 50, 51, 52, 53]

    y_future, X_future = make_future()

    for SID in station_IDs:
         y_future.insert(1,SID, model_training_results[SID].get_prediction(X_future).summary_frame()['mean'], True)


    y_future= y_future.drop('hourly_counts',axis=1)
    ped_predicted = y_future.transpose()
    type(ped_predicted.index)
    return ped_predicted
    #ped_predicted.index.is_numeric()


########

def make_linelayer(df, color_array):
    #Inputs: df with [startlat, startlon, destlat, destlon] and font color as str([R,G,B]) - yes '[R,G,B]'
    #Plots lines between each line's [startlon, startlat] and [destlon, destlat]
    #Returns: pydeck LineLayer
    return pdk.Layer(
        type='LineLayer',
        data=df,
        getSourcePosition = '[startlon, startlat]',
        getTargetPosition = '[destlon, destlat]',
        getColor = color_array,
        getWidth = '5')

def make_pedlayer(df, color_array):
    return pdk.Layer(
        "HeatmapLayer",
        data=df,
        opacity=0.1,
        get_position=["centroid_x", "centroid_y"],
        aggregation="mean",
        get_weight="ped_rate")


@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_ped_station_data():
    ped_stations = pd.read_json("https://data.melbourne.vic.gov.au/resource/h57g-5234.json")
    ped_stations.set_index("sensor_id",inplace=True)
    return ped_stations

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_ped_data_current(ped_stations):
    ped_current = pd.read_json("https://data.melbourne.vic.gov.au/resource/d6mv-s43h.json")
    ped_current = ped_current.groupby('sensor_id')['total_of_directions'].sum().to_frame()
    ped_current = ped_stations[['latitude','longitude']].join(ped_current)
    return ped_current

#@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def predict_ped_rates():
    model_future = pickle_from_S3('modeled_future.p');
    start_date = dt.datetime.now().astimezone(pytz.timezone('Australia/Sydney'));
    start_date -= dt.timedelta(hours = 0, minutes = start_date.minute, seconds = start_date.second, microseconds = start_date.microsecond);
    end_date = start_date + dt.timedelta(hours = 26)
    start_date = start_date.replace(tzinfo=None)
    end_date = end_date.replace(tzinfo=None)
    index = ((model_future['ds']>start_date) & (model_future['ds']<end_date))
    future = model_future[index].reset_index(drop=True)
    future=future.drop('ds',axis=1)
    future=future.transpose().join(ped_stations)
    return future

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_model_results_data():
    bucket='walkwize'
    data_key = 'poisson.p'
    data_location = 's3://{}/{}'.format(bucket, data_key)
    #poisson_training_results = pd.read_pickle(data_location)
    return pd.read_pickle(data_location)



def get_map_bounds(gdf_nodes, route1, route2):
    #Inputs: node df, and two lists of nodes along path
    #Returns: Coordinates of smallest rectangle that contains all nodes
    max_x = -1000
    min_x = 1000
    max_y = -1000
    min_y = 1000

    for i in (route1 + route2):
        row = gdf_nodes.loc[i]
        temp_x = row['x']
        temp_y = row['y']

        max_x = max(temp_x, max_x)
        min_x = min(temp_x, min_x)
        max_y = max(temp_y, max_y)
        min_y = min(temp_y, min_y)

    return min_x, max_x, min_y, max_y

def nodes_to_lats_lons(nodes, path_nodes):
    #Inputs: node df, and list of nodes along path
    #Returns: 4 lists of source and destination lats/lons for each step of that path for LineLayer
    #S-lon1,S-lat1 -> S-lon2,S-lat2; S-lon2,S-lat2 -> S-lon3,S-lat3
    source_lats = []
    source_lons = []
    dest_lats = []
    dest_lons = []

    for i in range(0,len(path_nodes)-1):
        source_lats.append(nodes.loc[path_nodes[i]]['y'])
        source_lons.append(nodes.loc[path_nodes[i]]['x'])
        dest_lats.append(nodes.loc[path_nodes[i+1]]['y'])
        dest_lons.append(nodes.loc[path_nodes[i+1]]['x'])

    return (source_lats, source_lons, dest_lats, dest_lons)

def get_nodes(G, s, e):
    #Inputs: Graph,  source, end
    if s == '':
        #No address, default to Insight
        st.write('Source address not found. Defaulting to the Immigration Museum, Melbourne.')
        s = 'Immigration Museum'
        start_location = ox.utils_geo.geocode(s)
    else:
        try:
            start_location = ox.utils_geo.geocode(s + ' Melbourne, Australia')
        except:
            #No address found, default to Insight
            st.write('Source address not found. Defaulting to the Immigration Museum, Melbourne.')
            s = 'Immigration Museum'
            start_location = ox.utils_geo.geocode(s)

    if e == '':
        #No address, default to Fenway Park
        st.write('Destination address not found. Defaulting to the State Library of Victoria.')
        e = 'State Library of Victoria'
        end_location = ox.utils_geo.geocode(e)
    else:
        try:
            end_location = ox.utils_geo.geocode(e + ' Melbourne, Australia')
        except:
            #No address found, default to Insight
            st.write('Destination address not found. Defaulting to the State Library of Victoria.')
            e = 'State Library of Victoria'
            end_location = ox.utils_geo.geocode(e)

    #Get coordinates from addresses
    start_coords = (start_location[0], start_location[1])
    end_coords = (end_location[0], end_location[1])

    #Snap addresses to graph nodes
    start_node = ox.get_nearest_node(G, start_coords)
    end_node = ox.get_nearest_node(G, end_coords)
    return start_node, end_node

def calculate_routes(G, gdf_nodes, gdf_edges, start_node, end_node, factor):
    lengths = {};
    ped_rates = {};

    for row in gdf_edges.itertuples():
        u = getattr(row,'u')
        v = getattr(row,'v')
        key = getattr(row, 'key')
        length = getattr(row, 'length')
        ped_rate = getattr(row, 'ped_rate')
        lengths[(u,v,key)] = length
        ped_rates[(u,v,key)] = ped_rate

    optimized = {}
    if (factor == 0):
        for key in lengths.keys():
            temp = (int(lengths[key]))
            optimized[key] = temp
    else:
        for key in lengths.keys():
            temp = (int(lengths[key])*(1+int(ped_rates[key])*(factor*factor*factor/1000)))
            optimized[key] = temp

    #Generate new edge attribute
    nx.set_edge_attributes(G, optimized, 'optimized')

    #Path of nodes
    optimized_route = nx.shortest_path(G, start_node, end_node, weight = 'optimized')
    shortest_route = nx.shortest_path(G, start_node, end_node, weight = 'length')
    min_dist = nx.shortest_path_length(G, start_node, end_node, weight = 'length')
    #optimized_route
    #shortest_route
    shortest_length = 0;
    shortest_time = 0;
    shortest_people = 0;
    for i in range(len(shortest_route)-1):
        source, target = shortest_route[i], shortest_route[i+1]
         # #shortest_people += lengths[(source,target,0)]*(1/4000)*ped_rates[(source,target,0)]
        shortest_length += lengths[(source,target,0)]

    optimized_length = 0;
    optimized_time = 0;
    optimized_people = 0;
    for i in range(len(optimized_route)-1):
        source, target = optimized_route[i], optimized_route[i+1]
    # #     optimized_people += lengths[(source,target,0)]*(1/4000)*ped_rates[(source,target,0)]
    optimized_length += lengths[(source,target,0)]

    short_start_lat, short_start_lon, short_dest_lat, short_dest_lon = nodes_to_lats_lons(gdf_nodes, shortest_route)
    short_df = pd.DataFrame({'startlat':short_start_lat, 'startlon':short_start_lon, 'destlat': short_dest_lat, 'destlon':short_dest_lon})
    short_layer = make_linelayer(short_df, '[160,160,160]')

    #This finds the bounds of the final map to show based on the paths
    #min_x, max_x, min_y, max_y = get_map_bounds(gdf_nodes, shortest_route, optimized_route)
    #min_x, max_x, min_y, max_y = -37.82,-37.805,144.95, 144.97

    # # #These are lists of origin/destination coords of the paths that the routes take
    # opt_start_lat, opt_start_lon, opt_dest_lat, opt_dest_lon = nodes_to_lats_lons(gdf_nodes, optimized_route)
    # #
    # # #Move coordinates into dfs
    # opt_df = pd.DataFrame({'startlat':opt_start_lat, 'startlon':opt_start_lon, 'destlat': opt_dest_lat, 'destlon':opt_dest_lon})
    # #
    # # start_node_df = get_node_df(start_location)
    # optimized_layer = make_linelayer(opt_df, '[0,0,179]')
    #
    #


    d = {'Shortest Route (grey)': [round(shortest_length/1000,2),round(shortest_people)]}

    #d = {'Shortest Route (grey)': [round(shortest_length/1000,2),round(shortest_people)], 'Optimized Route (blue)': [round(optimized_length/1000,2),round(optimized_people)]}
    df = pd.DataFrame(data=d)
    df.rename(index={0:"Distance, in km: "},inplace=True)
    df.rename(index={1:"Total expected pedestrian interactions: "},inplace=True)
    layers=[short_layer];
    return df, layers


#################### RUN THE WEB APP ####################################
# While the user is getting set up, the webapp should run, and estimate for the next 24 hours.
# Grab pickled resources
ped_stations = get_ped_stations();
gdf_nodes = get_gdf_nodes();
gdf_edges = get_gdf_edges();
G = get_map_data();

trained_models = get_trained_models();

# Grab live conditions
ped_current = get_ped_data_current(ped_stations)
# Predict future conditions
ped_predicted = predict_ped_rates();

#### WEB APP ####
st.sidebar.title("WalkWize");
st.sidebar.markdown("*Take the path least traveled*");
st.sidebar.header("Let's plan your walk!");

input_start = st.sidebar.text_input('Where will you start?');
input_dest = st.sidebar.text_input('Where are you going?');
slider_future = st.sidebar.slider('Conditions in __ hours?',0,24)
slider_factor = st.sidebar.slider('How crowd averse are you? (0-10)',0,10)

st.sidebar.header("Location Suggestions:");
st.sidebar.markdown("*State Library of Victoria, Immigration Museum, Flagstaff Gardens, Vision Apartments*")

gdf_edges['ped_rate'] = interpolate.griddata(np.array(tuple(zip(ped_current['latitude'], ped_current['longitude']))),np.array(ped_current['total_of_directions']),np.array(tuple(zip(gdf_edges['centroid_y'], gdf_edges['centroid_x']))), method='cubic',rescale=False,fill_value=0)

#ped_layer = make_pedlayer(gdf_edges[['centroid_x','centroid_y','ped_rate']],COLOR_BREWER_RED)

submit = st.sidebar.button('Find route', key=1)

if not submit:
    map_data = ped_current[['latitude','longitude']];
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=-37.8125,
            longitude=144.96,
            zoom=13.5)))

else:
    with st.spinner('Routing'):
        start_node, end_node = get_nodes(G, input_start, input_dest)

        ped_station_locations = np.array(tuple(zip(ped_current['latitude'], ped_current['longitude'])));

        if slider_future == 0:
            ped_station_values = np.array(ped_current['total_of_directions']);
        else:
            ped_station_locations = np.array(tuple(zip(ped_current['latitude'], ped_current['longitude'])));
            ped_station_values = np.array(ped_predicted[[slider_future]])

        edge_centroids = np.array(tuple(zip(gdf_edges['centroid_y'], gdf_edges['centroid_x'])));
        gdf_edges['ped_rate'] = interpolate.griddata(ped_station_locations,ped_station_values,edge_centroids, method='cubic',rescale=False,fill_value=0);

        df, layers = calculate_routes(G, gdf_nodes, gdf_edges, start_node, end_node, slider_factor)

        map_data = ped_current[['latitude','longitude']];
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=-37.8125,
                longitude=144.96,
                zoom=13.5,
            ),
            layers=[layers]
        ))

        st.table(df)
