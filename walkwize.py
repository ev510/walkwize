import datetime as dt
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pickle
import psycopg2
import pydeck as pdk
import scipy
import scipy.interpolate as interpolate
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st
import timeit

from patsy import dmatrices
#from sqlalchemy import create_engine
#from sqlalchemy_utils import database_exists, create_database






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

def get_ped_predicted():
	#[df_test, df_train, poisson_training_results, nb2_training_results,y_train,y_test,X_train,X_test] = pickle.load( open( "poisson.p", "rb" ) )#

	[poisson_training_results] = pickle.load( open( "poisson.p", "rb" ) )#
	station_IDs = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 17, 18, 19, 20, 21,
	            22, 23, 24, 26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 40, 41, 42, 43,
	            44, 45, 46, 47, 48, 49, 50, 51, 52, 53]

	y_future, X_future = make_future()

	for SID in station_IDs:
	 	y_future.insert(1,SID, poisson_training_results[SID].get_prediction(X_future).summary_frame()['mean'], True)

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
def get_map_data():
	#Returns: map as graph from graphml
	#Cached by Streamlit
	#G = ox.graph_from_bbox(-37.8061,-37.8200,144.9769, 144.9569, network_type='walk')
	G = ox.graph_from_bbox(-37.8000,-37.8250,144.9800, 144.9500, network_type='walk')

	gdf_nodes, gdf_edges = ox.utils_graph.graph_to_gdfs(G, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True)
	gdf_edges['centroid_x'] = gdf_edges.apply(lambda r: r.geometry.centroid.x, axis=1)
	gdf_edges['centroid_y'] = gdf_edges.apply(lambda r: r.geometry.centroid.y, axis=1)

	return G, gdf_nodes, gdf_edges

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_ped_station_data():
	ped_stations = pd.read_json("https://data.melbourne.vic.gov.au/resource/h57g-5234.json")
	ped_stations.set_index("sensor_id",inplace=True)

	return ped_stations

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_ped_data_current():
	ped_current = pd.read_json("https://data.melbourne.vic.gov.au/resource/d6mv-s43h.json")
	ped_current = ped_current.groupby('sensor_id')['total_of_directions'].sum().to_frame()
	ped_current = ped_current.join(ped_stations[['latitude','longitude']])


	return ped_current

def get_ped_historic_data():
	pass

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
	#S-lon1,S-lat1 -> S-lon2,S-lat2; S-lon2,S-lat2 -> S-lon3,S-lat3...
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

def source_to_dest(G, gdf_nodes, gdf_edges, s, e):
	#Inputs: Graph, nodes, edges, source, end, distance to walk, pace = speed, w2 bool = avoid busy roads

	if s == '':
		#No address, default to Insight
		st.write('Source address not found, defaulting...')
		s = '440 Elizabeth St, Melbourne VIC 3000, Australia'
		start_location = ox.utils_geo.geocode(s)
	else:
		try:
			start_location = ox.utils_geo.geocode(s + ' Melbourne, Australia')
		except:
			#No address found, default to Insight
			st.write('Source address not found, defaulting...')
			s = '440 Elizabeth St, Melbourne VIC 3000, Australia'
			start_location = ox.utils_geo.geocode(s)

	if e == '':
		#No address, default to Fenway Park
		st.write('Destination address not found, defaulting...')
		e = '1 Spring St, Melbourne VIC 3000, Australia'
		end_location = ox.utils_geo.geocode(e)
	else:
		try:
			end_location = ox.utils_geo.geocode(e + ' Melbourne, Australia')
		except:
			#No address found, default to Insight
			st.write('Destination address not found, defaulting...')
			e = '1 Spring St, Melbourne VIC 3000, Australia'
			end_location = ox.utils_geo.geocode(e)

	#Get coordinates from addresses
	start_coords = (start_location[0], start_location[1])
	end_coords = (end_location[0], end_location[1])

	#Snap addresses to graph nodes
	start_node = ox.get_nearest_node(G, start_coords)
	end_node = ox.get_nearest_node(G, end_coords)

	lengths = {}
	ped_rates = {}
	factor = 1
	for row in gdf_edges.itertuples():
		u = getattr(row,'u')
		v = getattr(row,'v')
		key = getattr(row, 'key')
		length = getattr(row, 'length')
		ped_rate = getattr(row, 'ped_rate')
		lengths[(u,v,key)] = length
		ped_rates[(u,v,key)] = ped_rate

	optimized = {}
	for key in lengths.keys():
		#temp = int(lengths[key])
		temp = (int(lengths[key])*(int(ped_rates[key]+1)))
		optimized[key] = temp





	#Generate new edge attribute
	nx.set_edge_attributes(G, optimized, 'optimized')

	#Path of nodes
	optimized_route = nx.shortest_path(G, start_node, end_node, weight = 'optimized')


	shortest_route = nx.shortest_path(G, start_node, end_node, weight = 'length')
	short_start_lat, short_start_lon, short_dest_lat, short_dest_lon = nodes_to_lats_lons(gdf_nodes, shortest_route)
	short_df = pd.DataFrame({'startlat':short_start_lat, 'startlon':short_start_lon, 'destlat': short_dest_lat, 'destlon':short_dest_lon})
	short_layer = make_linelayer(short_df, '[160,160,160]')

	#This finds the bounds of the final map to show based on the paths
	min_x, max_x, min_y, max_y = get_map_bounds(gdf_nodes, shortest_route, optimized_route)

	#These are lists of origin/destination coords of the paths that the routes take
	opt_start_lat, opt_start_lon, opt_dest_lat, opt_dest_lon = nodes_to_lats_lons(gdf_nodes, optimized_route)


	#Move coordinates into dfs
	opt_df = pd.DataFrame({'startlat':opt_start_lat, 'startlon':opt_start_lon, 'destlat': opt_dest_lat, 'destlon':opt_dest_lon})

	COLOR_BREWER_RED = [[255,247,236],[254,232,200],
		[253,212,158],[253,187,132],
		[252,141,89],[239,101,72],
		[215,48,31],[179,0,0],[127,0,0]]

	start_node_df = get_node_df(start_location)
	optimized_layer = make_linelayer(opt_df, '[0,0,179]')
	#ped_layer = make_pedlayer(ped_current,COLOR_BREWER_RED)
	ped_layer = make_pedlayer(gdf_edges[['centroid_x','centroid_y','ped_rate']],COLOR_BREWER_RED)


	# type(gdf_edges)
	# type(ped_current)
	#
	# gdf_edges
	st.pydeck_chart(pdk.Deck(
		map_style="mapbox://styles/mapbox/light-v9",
		initial_view_state=pdk.ViewState(latitude = -37.81375, longitude = 144.9669, zoom=13.5),
		layers=[short_layer, optimized_layer, ped_layer]))


	st.write('The path of shortest distance is shown in grey. The path of least contact is shown in blue.')
	return



#################### RUN THE WEB APP ####################################
# While the user is getting set up, the webapp should run, and estimate for the next 24 hours.
# Start with historic trends icon_layer
# Then implement model based current trends (a different model?)

#import model parameters

G, gdf_nodes, gdf_edges= get_map_data()
ped_stations = get_ped_station_data()
ped_current = get_ped_data_current()
ped_predicted = get_ped_predicted()
ped_predicted.index.name = 'sensor_id'
ped_predicted = pd.concat([ped_predicted,ped_stations[['latitude','longitude']]],axis=1, join="inner")

st.sidebar.title("WalkWize");
st.sidebar.markdown("*Take the path least traveled*");
st.sidebar.header("Let's plan your walk!");

input1 = st.sidebar.text_input('Where will you start?');
input2 = st.sidebar.text_input('Where are you going?');
slider = st.sidebar.slider('Conditions in __ hours?',0,24)

#date = st.sidebar.date_input('When you you want to leave?',  max_value=dt.datetime(2020, 12, 31, 0, 0));
#time = st.sidebar.time_input('What time do you want to leave?', value=None, key=None);

gdf_edges['ped_rate'] = interpolate.griddata(np.array(tuple(zip(ped_current['latitude'], ped_current['longitude']))),np.array(ped_current['total_of_directions']),np.array(tuple(zip(gdf_edges['centroid_y'], gdf_edges['centroid_x']))), method='cubic',rescale=False,fill_value=0)




# COLOR_BREWER_RED is not activated, default color range is used
COLOR_BREWER_RED = [[255,247,236],[127,0,0]]
ped_layer = make_pedlayer(gdf_edges[['centroid_x','centroid_y','ped_rate']],COLOR_BREWER_RED)


submit = st.sidebar.button('Find route', key=1)
if not submit:
	st.pydeck_chart(pdk.Deck(
		map_style="mapbox://styles/mapbox/light-v9",
		initial_view_state=pdk.ViewState(latitude = -37.81375, longitude = 144.9669, zoom=13.5),
		layers=[ped_layer]))
else:
	with st.spinner('Routing...'):
		if slider == 0:
			gdf_edges['ped_rate'] = interpolate.griddata(np.array(tuple(zip(ped_current['latitude'], ped_current['longitude']))),np.array(ped_current['total_of_directions']),np.array(tuple(zip(gdf_edges['centroid_y'], gdf_edges['centroid_x']))), method='cubic',rescale=False,fill_value=0)

		else:
			gdf_edges['ped_rate'] = interpolate.griddata(np.array(tuple(zip(ped_predicted['latitude'], ped_predicted['longitude']))),np.array(ped_predicted[ped_predicted.columns[slider]]),np.array(tuple(zip(gdf_edges['centroid_y'], gdf_edges['centroid_x']))), method='cubic',rescale=False,fill_value=0)

		st.markdown(ped_predicted.columns[slider])
		gdf_edges['ped_rate'] = gdf_edges['ped_rate'].clip(lower=0)
		source_to_dest(G, gdf_nodes, gdf_edges, input1, input2)

poisson_predictions = poisson_training_results[SID].get_prediction(X_test[SID]).summary_frame()['mean']
nb2_predictions = nb2_training_results[SID].get_prediction(X_test[SID]).summary_frame()['mean']


#slider = st.slider('How much do you want to avoid people?',0,24)
#timeframe = st.radio("Using what paradigm?",('Pre-COVID', 'Current'))


###### Generating figures #######
###### Not used in webapp #######
SID =4

# Sample time-series data. Consider highlighting zero-inflatedness.
a = plt.figure(figsize=(4,4));
axes = a.add_axes([.2, .2, .7, .7]);
axes.plot(y_train[SID],'.',label='training data');
#axes.plot(pd.to_datetime(train_datetime),predictions,'.')
axes.plot(y_test[SID],'.',label='testing data');
axes.plot(poisson_predictions,'.',label='poisson prediction');
#axes.plot(nb2_predictions,'.',label='nb2');
axes.set_xlim(737014, 737018);
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
axes.plot(ped_predicted2.transpose()[2], label='forecast')	;
import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%H:00')
axes.xaxis.set_major_formatter(myFmt)
axes.legend(fancybox = True, framealpha=0);
b.autofmt_xdate()
b.savefig('predict.png', transparent = True,dpi=600)
b
