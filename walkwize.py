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
import s3fs
import boto3

from patsy import dmatrices
#from sqlalchemy import create_engine
#from sqlalchemy_utils import database_exists, create_database


def pickle_from_S3(key):
    # key = 'poisson.p'
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

def get_node_df(location):
	#Inputs: location as tuple of coords (lat, lon)
	#Returns: 1-line dataframe of location for map layer
	return pd.DataFrame({'lat':[location[0]], 'lon':[location[1]]})

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_ped_data_current():
	ped_current = pd.read_json("https://data.melbourne.vic.gov.au/resource/d6mv-s43h.json")
	ped_current = ped_current.groupby('sensor_id')['total_of_directions'].sum().to_frame()
	ped_current = ped_stations[['latitude','longitude']].join(ped_current)
	return ped_current


def make_prophet_future():
    # Creating the next 24 hours of AUSTRALIA time
    now = dt.datetime.utcnow()
    # Round to the next hour
    now -= dt.timedelta(hours = -10-1, minutes = now.minute, seconds = now.second, microseconds = now.microsecond)
    # Create next 24 hours
    prophet_future = pd.date_range(now, now+dt.timedelta(hours=24),freq='h').to_frame()
    prophet_future = prophet_future.rename(columns={0:'ds'}).reset_index().drop('index',axis=1);
    return prophet_future

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def predict_ped_rates(trained_models, ped_stations):
	future = make_prophet_future()
	for SID in ped_stations.index:
		future[SID] = trained_models[SID].predict(future)['yhat'].clip(lower=0)
	future=future.drop('ds',axis=1)
	future=future.transpose().join(ped_stations)
	return future

#### MAPPING ####
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

def make_pedlayer(df):
	#Inputs: df with [centroid_x, centroid_y, ped]
	#Generates heatmap layer
	#Returns: pydeck LineLayer
	return pdk.Layer(
		"HeatmapLayer",
		data=df,
		opacity=0.1,
		get_position=["centroid_x", "centroid_y"],
		aggregation="mean",
		get_weight="ped_rate")

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

def calculate_routes(G, gdf_nodes, gdf_edges, s, e, factor):
	#Inputs: Graph,  source, end

	if s == '':
		#No address, default to Insight
		st.write('Source address not found, defaulting to Immigration Museum')
		s = 'Immigration Museum'
		start_location = ox.utils_geo.geocode(s)
	else:
		try:
			start_location = ox.utils_geo.geocode(s + ' Melbourne, Australia')
		except:
			#No address found, default to Insight
			st.write('Source address not found, defaulting to Immigration Museum')
			s = 'Immigration Museum'
			start_location = ox.utils_geo.geocode(s)

	if e == '':
		#No address, default to Fenway Park
		st.write('Destination address not found, defaulting to State Library')
		e = 'State Library of Victoria'
		end_location = ox.utils_geo.geocode(e)
	else:
		try:
			end_location = ox.utils_geo.geocode(e + ' Melbourne, Australia')
		except:
			#No address found, default to Insight
			st.write('Destination address not found, defaulting to State Library')
			e = 'State Library of Victoria'
			end_location = ox.utils_geo.geocode(e)

	#Get coordinates from addresses
	start_coords = (start_location[0], start_location[1])
	end_coords = (end_location[0], end_location[1])

	#Snap addresses to graph nodes
	start_node = ox.get_nearest_node(G, start_coords)
	end_node = ox.get_nearest_node(G, end_coords)

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



	short_start_lat, short_start_lon, short_dest_lat, short_dest_lon = nodes_to_lats_lons(gdf_nodes, shortest_route)
	short_df = pd.DataFrame({'startlat':short_start_lat, 'startlon':short_start_lon, 'destlat': short_dest_lat, 'destlon':short_dest_lon})
	short_layer = make_linelayer(short_df, '[160,160,160]')

	#This finds the bounds of the final map to show based on the paths
	#min_x, max_x, min_y, max_y = get_map_bounds(gdf_nodes, shortest_route, optimized_route)
	min_x, max_x, min_y, max_y = -37.82,-37.805,144.95, 144.97

	#These are lists of origin/destination coords of the paths that the routes take
	opt_start_lat, opt_start_lon, opt_dest_lat, opt_dest_lon = nodes_to_lats_lons(gdf_nodes, optimized_route)


	#Move coordinates into dfs
	opt_df = pd.DataFrame({'startlat':opt_start_lat, 'startlon':opt_start_lon, 'destlat': opt_dest_lat, 'destlon':opt_dest_lon})

	start_node_df = get_node_df(start_location)
	optimized_layer = make_linelayer(opt_df, '[0,0,179]')
	ped_layer = make_pedlayer(gdf_edges[['centroid_x','centroid_y','ped_rate']])

	st.pydeck_chart(pdk.Deck(
		map_style="mapbox://styles/mapbox/light-v9",
		initial_view_state=pdk.ViewState(latitude = -37.8125, longitude = 144.96, zoom=13.5),
		layers=[short_layer, optimized_layer, ped_layer]))

	min_x, max_x, min_y, max_y = -37.82,-37.805,144.95, 144.97
	st.write('The path of shortest distance is shown in grey. The path of least contact is shown in blue.')
	return

#################### RUN THE WEB APP ####################################

# While the user is getting set up, the webapp should run, and estimate for the next 24 hours.
# Grab pickled resources
ped_stations = get_ped_stations();
gdf_nodes = get_gdf_nodes();
gdf_edges = get_gdf_edges();
G = get_map_data();
trained_models = get_trained_models();

# Grab live conditions
ped_current = get_ped_data_current()

len(ped_stations)
# Predict future conditions
ped_predicted = predict_ped_rates(trained_models, ped_stations);

#### WEB APP ####
st.sidebar.title("WalkWize");
st.sidebar.markdown("*Take the path least traveled*");
st.sidebar.header("Let's plan your walk!");

input_start = st.sidebar.text_input('Where will you start?');
input_dest = st.sidebar.text_input('Where are you going?');
slider_future = st.sidebar.slider('Conditions in __ hours?',0,24)
slider_factor = st.sidebar.slider('How crowd averse are you? (1-10)',0,10)

st.sidebar.header("Location Suggestions:");
st.sidebar.markdown("*State Library of Victoria, Immigration Museum, Flagstaff Gardens, Vision Apartments*")

gdf_edges['ped_rate'] = interpolate.griddata(np.array(tuple(zip(ped_current['latitude'], ped_current['longitude']))),np.array(ped_current['total_of_directions']),np.array(tuple(zip(gdf_edges['centroid_y'], gdf_edges['centroid_x']))), method='cubic',rescale=False,fill_value=0)

gdf_edges.to_pickle("gdf.p")
ped_layer = make_pedlayer(gdf_edges[['centroid_x','centroid_y','ped_rate']])

submit = st.sidebar.button('Find route', key=1)
st.markdown("The current time in Melbourne: {}".format(dt.datetime.utcnow()-dt.timedelta(hours = -10)))

if not submit:
	st.pydeck_chart(pdk.Deck(
		map_style="mapbox://styles/mapbox/light-v9",
		initial_view_state=pdk.ViewState(latitude = -37.8125, longitude = 144.96, zoom=13.5),
		layers=[ped_layer]))
else:
	with st.spinner('Routing...'):
		if slider_future == 0:
			gdf_edges['ped_rate'] = interpolate.griddata(np.array(tuple(zip(ped_current['latitude'], ped_current['longitude']))),np.array(ped_current['total_of_directions']),np.array(tuple(zip(gdf_edges['centroid_y'], gdf_edges['centroid_x']))), method='cubic',rescale=False,fill_value=0)

		else:
			gdf_edges['ped_rate'] = interpolate.griddata(np.array(tuple(zip(ped_predicted['latitude'], ped_predicted['longitude']))),np.array(ped_predicted[[slider_future]]),np.array(tuple(zip(gdf_edges['centroid_y'], gdf_edges['centroid_x']))), method='cubic',rescale=False,fill_value=0)

		st.markdown(ped_predicted.columns[slider_future])
		gdf_edges['ped_rate'] = gdf_edges['ped_rate'].clip(lower=0)
		calculate_routes(G, gdf_nodes, gdf_edges, input_start, input_dest, slider_factor)
