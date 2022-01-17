import streamlit as st
import pandas as pd
import datetime
import numpy as np
import missingno as mano
from datetime import timedelta
import json
import requests
import time
import random
from time import strftime
from time import gmtime
import calendar

import seaborn as sns
import matplotlib.pyplot as plt

from plotnine import *
from plotnine.data import economics
from plotnine import ggplot, aes, geom_line
import plotly.express as px 
import plotly.graph_objs as go
from scipy import stats
import altair as alt

from sklearn import preprocessing
from sklearn.cluster import KMeans

import sklearn.cluster as cluster
from sklearn.decomposition import PCA

#Time stamp fonction
def get_date(dt):
    return dt.date()
def get_year(dt):
    return dt.year
def get_month(dt):
    return dt.month
def get_day(dt):
    return dt.day
def get_weekday(dt):
    return dt.weekday() # .weekday() is a method
def get_hour(dt):
    return dt.hour
def get_min(dt):
    return dt.minute
def get_sec(dt):
    return dt.second
def get_time(dt):
    return dt.time()

def week_number(df,col):
    df["week_number"] = df[col].apply(lambda x: x.strftime("%V"))
    return df

def get_month_and_month_name(df,col):
    import calendar
    df["month"] = df[col].map(get_month)
    df["month_letter"] = df["month"].apply(lambda x:calendar.month_name[x])
    return df

def get_day_and_day_name(df,col):
    import calendar
    df["day"] = df[col].map(get_weekday)
    df['day_letter'] = df[col].apply(lambda x: datetime.datetime.strftime(x, '%A'))
    return df
#Data process fonction
def tranform_ms_to_min(df):
    df["endTime"] = df["endTime"].map(pd.to_datetime)
    df["min_played"] = df["msPlayed"].apply(lambda x: x/1000)
    df["min_played"] = df["min_played"].apply(lambda x: x/60)
    return df

def get_date_and_time(df):
    df["ending_time"] = df["ending_date_time"].map(get_time)
    df["date"] = df["ending_date_time"].map(get_date)
    df["date"] = df["date"].map(pd.to_datetime)
    return df

def date_generation(df,starting_date,ending_date):
    #df = pd.DataFrame()
    df["date"] = pd.date_range(start=starting_date,end=ending_date)
    df["min_played"] = 0
    df["msPlayed"] = 0
    df["patern"] = "date_ajoute"
    df["ending_date_time"] = df["date"]
    df["ending_time"] = df["date"].map(get_time)
    return df

def find_missing_day(df1,df2,col):
    fusion = df1.merge(df2,on=col,how="left")
    date_a_ajoute = fusion.loc[(pd.isnull(fusion['ending_date_time_y'])== True)]
    date_a_ajoute.drop(["patern","ending_date_time_y","artistName","trackName","msPlayed_y","min_played_y","ending_time_y"],axis=1,inplace=True)
    date_a_ajoute.rename(columns={"min_played_x":"min_played","msPlayed_x":"msPlayed","ending_date_time_x":"ending_date_time","ending_time_x":"ending_time"},inplace=True)
    date_a_ajoute["date"] = date_a_ajoute["date"].map(pd.to_datetime)
    return date_a_ajoute

#Graphics process fonctions
#Graph 1 : Number of minute played per week during a year
def min_played_per_week(df,col):
    group_week_data = df.groupby(df[col]).min_played.agg(sum).reset_index()
    group_week_data[col] = group_week_data[col].astype(int)
    return group_week_data

#Graph 2 : SPOTIFY USAGE ON A MONTH
def month_listen_habit(df,month_option):
	spotify_month = df[df["month_letter"]==month_option]
	return spotify_month
def line_plot_grp_by(df,gb_col,agg_col,title):
	plot_month_habit = alt.Chart(df.groupby([gb_col]).count()[agg_col].reset_index()).mark_line().encode(x=gb_col,y=agg_col).properties(title=title)
	return plot_month_habit

#Graph 3 : Number of hour played for each day on a year
def spotify_number_of_hour_played_by_days(df,gb1_value,gb2_value,agg_value):
    number_of_hour_played_per_day = df.groupby([gb1_value, gb2_value]).agg({agg_value: ['sum']})
    number_of_hour_played_per_day.columns =  ['sum_played_hour']
    number_of_hour_played_per_day = number_of_hour_played_per_day.reset_index()
    return number_of_hour_played_per_day
def plot_pie(df,value,name):
    pie_fig = px.pie(df,values=value, names=name)
    pie_fig.update_traces(textposition='inside', textinfo='percent+label')
    return pie_fig
def day_listen_habit(df,day_option):
	spotify_full_day = df[df["day_letter"]==day_option]
	return spotify_full_day
def plot_line(df,x_axis,y_axis,graph_title):
	line_chart_conso_per_day = alt.Chart(df).mark_line().encode(x=x_axis, y=y_axis).properties(title=graph_title)
	return line_chart_conso_per_day

#GRAPH 4 : Spotify usage on a day
def heat_map_grp_by(df,gb_val,agg_val):
    gb_heat_map, ax = plt.subplots()
    ax = sns.set(rc={'figure.figsize':(20.7,8.27)})
    gb_heat_map = sns.heatmap(df.groupby([gb_val]).count()[[agg_val]], ax=ax)
    return gb_heat_map
#GRAPH 5 : How many music are played in a day
def spotify_number_of_songs_per_day(df,gb_col):
    spotify_song_per_day = df.groupby(df[gb_col]).trackName.agg(list).reset_index()
    spotify_song_per_day["number_of_song_per_day"] = spotify_song_per_day["trackName"].apply(lambda x : len(x))
    spotify_song_per_day["number_of_song_per_day"] = spotify_song_per_day["number_of_song_per_day"].apply(lambda x : x if x!=1 else 0)
    spotify_song_per_day  = spotify_song_per_day[spotify_song_per_day["number_of_song_per_day"]!=0]
    spotify_song_per_day.sort_values(by=['number_of_song_per_day'],ascending=False,inplace=True)
    spotify_song_per_day.drop(['trackName'],axis=1,inplace=True)
    return spotify_song_per_day

def plot_average_number_of_songs_listened_in_a_day(df,xcol,ycol,mean_col):
    number_of_song_per_day,ax = plt.subplots(figsize=(15,8))
    ax.scatter(x=xcol, y = ycol);
    ax.set(title="Maximum number of songs played in a day",xlabel="Date",ylabel="number of song per day");
    ax.axhline(df[mean_col].mean(), linestyle="-", color="r");
    return number_of_song_per_day, ax
#GRAPH 6 : Most listen artist
def most_listen(df,gb_value,sort_value):
	most_listen_artist = df.groupby(df[gb_value]).min_played.agg(sum).reset_index()
	most_listen_artist.sort_values(by=[sort_value],inplace=True, ascending=False)
	return most_listen_artist
def plot_bar(df,x_axis,y_axis,title):
	most_listen_artist=px.bar(df,x=x_axis,y=y_axis, orientation='h',title=title)
	most_listen_artist.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})
	return most_listen_artist
def artist_with_most_albums (df,gb_col,sort_value):
	group_artist_number_song_data = df.groupby(df[gb_col]).trackName.agg(list).reset_index()
	group_artist_number_song_data["trackName_unique"] = group_artist_number_song_data["trackName"].apply(lambda x : set(x))
	group_artist_number_song_data["number_songs"] = group_artist_number_song_data["trackName_unique"].apply(lambda x: len(x))
	group_artist_number_song_data.sort_values(by=[sort_value],inplace=True,ascending=False)
	return group_artist_number_song_data
#GRAPH 8 : Correlation between the hour spend on an artist and the number of song he released

def artist_name_and_number_song_relised(df,gb_coll,apply_col):
	GB_artist_number_of_song = df.groupby(df[gb_coll]).trackName.agg(list).reset_index()
	GB_artist_minute_played = df.groupby(df[gb_coll]).min_played.agg(sum).reset_index()
	artist_name_number_song_and_minute_listen = GB_artist_minute_played.merge(GB_artist_number_of_song,on=gb_coll,how="left")

	artist_name_number_song_and_minute_listen["trackName_unique"] = artist_name_number_song_and_minute_listen[apply_col].apply(lambda x : set(x))
	artist_name_number_song_and_minute_listen["number_songs"] = artist_name_number_song_and_minute_listen["trackName_unique"].apply(lambda x: len(x))
	artist_name_number_song_and_minute_listen.sort_values(by=['min_played'],inplace=True,ascending=False)
	return artist_name_number_song_and_minute_listen

def scratter_plot(df,x_axis,y_axis,title):
	scatter_plot_number_music_artist_corr = alt.Chart(df,title=title).mark_circle().encode(
     x=x_axis, y=y_axis)
	return scatter_plot_number_music_artist_corr

#Graph 9 : Music style analysis
def most_listen_music_type(df,col):
	df_library_spotify_genre_plus_ecoute = df[col].value_counts().reset_index()
	df_library_spotify_genre_plus_ecoute.rename(columns={"index":"music_type","genres":"number_of_music"},inplace=True)
	return df_library_spotify_genre_plus_ecoute

#Graph 10 :Audio feature analysis
def audio_features_mean(df,index):
	audio = df.columns[index:]
	mean_audio_feature = df[audio].mean()
	mean_audio_feature = mean_audio_feature.reset_index()
	mean_audio_feature.rename(columns={"index":"feature",0:"mean"},inplace=True)
	return mean_audio_feature
#Graph 11 : music variation analysis
def audio_features_std(df,index):
	audio = df.columns[index:]
	std_audio_feature = df_library_spotify_genre_mood[audio].std()
	std_audio_feature = std_audio_feature.reset_index()
	std_audio_feature.rename(columns={"index":"feature",0:"standar_dev"},inplace=True)
	return std_audio_feature
def plot_mean_and_std_music_feature(df,x_axis,y1_axis,y2_axis,title):
	mean_std = go.Figure(
	    data=[
	        go.Bar(
	            name="Mean",
	            x=df[x_axis],
	            y=df[y1_axis],
	            offsetgroup=0),
	        go.Bar(
	            name="Standar deviation",
	            x=df[x_axis],
	            y=df[y2_axis],
	            offsetgroup=1)],
	    layout=go.Layout(
	        title=title,
	        yaxis_title="Mean and standar deviation"))
	return mean_std

#Graph 12: Correlation analysis
def heat_map(df):
	audio_feature_corr_heatmap, ax = plt.subplots()
	ax = sns.set(rc={'figure.figsize':(20.7,8.27)})
	sns.heatmap(df.corr(), ax=ax)
	return audio_feature_corr_heatmap
def correlation_scatter(df,x_axis,y_axis,title):
	scatter_corr, ax = plt.subplots()
	ax = sns.set(rc={'figure.figsize':(20.7,8.27)})
	sns.scatterplot(x=x_axis, y=y_axis, data=df).set(title='Correlation between Valence and Danceability')
	return scatter_corr

#Graph 13 : What kind of music Im listening
def plot_polar(df,r_value,theta_value,title):
	polar = px.line_polar(df, r=r_value, theta=theta_value, line_close=True,template="plotly_dark",title = title)
	polar.update_traces(fill='toself')
	return polar
#GRAPH 15 : number song per cluster
def number_song_per_cluster(df,gb_value,apply_value):
	cluster = df.groupby(df[gb_value]).song_name.agg(list).reset_index()
	cluster["nbre_chanson_par cluster"] = cluster[apply_value].apply(lambda x: len(x))
	return cluster

#read data source
spotify = pd.read_csv('spotify_streaming_history.csv')
#Checking missing values
mano.bar(spotify) #Ther is no missing values

#Data processing

#get listen minute from listen milliseconds
tranform_ms_to_min(spotify)
spotify.rename(columns={"endTime":"ending_date_time"},inplace=True)
#Seperate de time and the date from datetime
get_date_and_time(spotify)
#All the days that i did not use spotify are not taken in consideration, so our date line isnot continious. I'm generating all the date in order to have a continious representation of our data, and highlight some paterns
all_date = pd.DataFrame()
date_generation(all_date,"2020-12-15","2021-12-15")
#we try to find all the missing date in spotify data frame
date_a_ajoute = find_missing_day(all_date,spotify,"date")
#We add on a new dataset spotify data and missing date, then we sort all the dataframe by date 
spotify_full_date = pd.concat([spotify,date_a_ajoute])
spotify_full_date.sort_values(by=['date'],inplace=True)

#We are getting week number on a year and hours,month and day from listening date. Month and day are also displayed by the name !
week_number(spotify_full_date,"date")
spotify_full_date["hours"] = spotify_full_date["ending_date_time"].map(get_hour)
get_month_and_month_name(spotify_full_date,"date")
get_day_and_day_name(spotify_full_date,"date")

#presentation images
""" [![Github](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Siva-chane/streamlit) &nbsp[![Linkedin](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/siva-chanemougam-701589143/) """
from PIL import Image
image = Image.open('logo_spotify.png')
st.image(image)
st.title("Personal data analysis")
#Basic Metrics on our data set 
col1, col2, col3 = st.columns(3)
col1.metric("Number of artist", len(spotify_full_date["artistName"].unique()))
col2.metric("Number of songs", len(spotify_full_date["trackName"].unique()))
col3.metric("listening hours ", int(spotify_full_date["min_played"].sum()/60))
st.write("We are going to analyse my musique streaming habit and try to highligt some insight!")

#GRAPH 1 : Number of minute played per week during a year
group_week_data = min_played_per_week(spotify_full_date,'week_number')
min_played_per_week = alt.Chart(group_week_data).mark_bar(size=10).encode(x='week_number', y='min_played', color='min_played').properties(title='Number of minute played per week during a year')
st.altair_chart(min_played_per_week, use_container_width=True)

#GRAPH 2 : SPOTIFY USAGE ON A MONTH
st.subheader('Spotify usage on month by month')
month_option = st.sidebar.selectbox("select a month",('January', 'February', 'March', 'April', 'May', 'June','July', 'August', 'September', 'October', 'November',"December"))
spotify_mois = month_listen_habit(spotify_full_date,month_option)
listen_per_month = line_plot_grp_by(spotify_mois,"date","min_played","Spotify usage on %s" % month_option)
st.altair_chart(listen_per_month, use_container_width=True)

#GRAPH 3 : Spotify usage on every week day
st.subheader('Spotify usage per week')
#Sidebar option
day_option = st.sidebar.selectbox('select a month',('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday'))
#pie chart view
played_hour_per_day = spotify_number_of_hour_played_by_days(spotify_full_date,"day_letter","hours","min_played")
day_use_pie_view = plot_pie(played_hour_per_day,'sum_played_hour','day_letter')
#line chart viex
spotify_day = day_listen_habit(spotify_full_date,day_option)
listen_habit_on_week_day = spotify_number_of_hour_played_by_days(spotify_day,"day_letter","hours","min_played")
line_plot_conso_per_week_day = plot_line(listen_habit_on_week_day,"hours","sum_played_hour","Spotify usage on %s" %day_option)
#Select view
view_style = st.radio("Choose your view",('Pie chart', 'Line chart'))
if view_style == 'Pie chart':
     st.write(day_use_pie_view)
else:
	st.altair_chart(line_plot_conso_per_week_day, use_container_width=True)

#GRAPH 4 : Spotify usage on a day
st.subheader('Spotify usage on a day')
#Line
spotify_usage_day_line_chart = line_plot_grp_by(spotify_full_date,"hours","min_played","Average use of Spotify on a day")
#heat map
usage_per_day_heat_map = heat_map_grp_by(spotify_full_date,"hours","min_played")
view_style_2 = st.radio("Choose your view",('Line chart', 'Heatmap'))

if view_style_2 == 'Line chart':
     st.write(spotify_usage_day_line_chart)

else:
	st.write(usage_per_day_heat_map)

#GRAPH 5 : How many music are played in a day
st.subheader('How many musics are played in a day')
spotify_song_per_day = spotify_number_of_songs_per_day(spotify_full_date,"date")
#Calculating the average, min and max number of music played per day

mean_music_per_day = int(spotify_song_per_day["number_of_song_per_day"].mean())
max_music_per_day = spotify_song_per_day["number_of_song_per_day"].max()
min_music_per_day = spotify_song_per_day["number_of_song_per_day"].min()

col1, col2, col3 = st.columns(3)
col1.metric("Mean Number of music played on a day", mean_music_per_day)
col2.metric("Max Number of music played on a day", max_music_per_day, "+")
col3.metric("Min Number of music played on a day", min_music_per_day, "-")

number_of_song_per_day, ax = plot_average_number_of_songs_listened_in_a_day(spotify_song_per_day, spotify_song_per_day["date"],spotify_song_per_day["number_of_song_per_day"],"number_of_song_per_day")
st.write(number_of_song_per_day,ax)

#GRAPH 6 : Most listen artist
st.subheader('Music Listen Artist')
dataframe_head_number = st.sidebar.select_slider("Select the number of row you want to display",options=[5, 10, 15,20])
col1, col2 = st.columns(2)

most_listen_artist = most_listen(spotify_full_date,"artistName","min_played")
most_listen_artist = most_listen_artist.head(dataframe_head_number)
artist_most_listen = plot_bar(most_listen_artist,'min_played','artistName',"Most listened Artist: first %s" %dataframe_head_number)

artist_with_most_albums = artist_with_most_albums(spotify_full_date,'artistName','number_songs')
artist_with_most_albums = artist_with_most_albums.head(dataframe_head_number)
number_music_per_artist = plot_bar(artist_with_most_albums,'number_songs','artistName',"Artist with the most album: first  %s" %dataframe_head_number)

col1.write(artist_most_listen)
col2.write(number_music_per_artist)

#GRAPH 7 : Most listen music
st.subheader('Most listened music')
#view per year
most_listen_music = most_listen(spotify_full_date,"trackName","min_played")
most_listen_music = most_listen_music.head(dataframe_head_number)
most_listen_music_full_year = plot_bar(most_listen_music,'min_played','trackName',"Most listened music in a full year")

#view par month
most_listen_music_per_month = most_listen(spotify_mois,"trackName","min_played")
most_listen_music_per_month = most_listen_music_per_month.head(dataframe_head_number)
most_listen_music_full_month = plot_bar(most_listen_music_per_month,'min_played','trackName',"Most listened music %s"%month_option)
#chose between anual or month view
view_style_3 = st.radio("Choose your view",('Annual view', 'month view'))

if view_style_3 == 'Annual view':
     st.write(most_listen_music_full_year)

else:
	st.write(most_listen_music_full_month)

#GRAPH 8 : Correlation between the hour spend on an artist and the number of song he released
st.subheader('Is there a correlation between the hour spend on an artist and the number of song he release ?')
artist_name_number_song_and_listen_minute = artist_name_and_number_song_relised(spotify_full_date,'artistName',"trackName")
corr_artiste_liseted_and_number_music_released = scratter_plot(artist_name_number_song_and_listen_minute,'min_played','number_songs',"Correlation between the hour spend on an artist and the number of song he release")
st.altair_chart(corr_artiste_liseted_and_number_music_released, use_container_width=True)

#GRAPH 9 : Music style analysis
st.subheader('Musique style analysis')
#Got this data frame by requesting Spotify API
df_library_spotify_genre_mood =  pd.read_csv("df_library_spotify_genre_mood (1).csv")
#bar chart view
spotify_most_listen_music_type = most_listen_music_type(df_library_spotify_genre_mood,"genres")
spotify_most_listen_music_type_bar_chart = spotify_most_listen_music_type.head(dataframe_head_number)
most_listen_type_music = plot_bar(spotify_most_listen_music_type_bar_chart,'number_of_music','music_type',"Most listen type of music")
#pie chart view
pie_fig_type_musique = plot_pie(spotify_most_listen_music_type,'number_of_music','music_type')
#interaction param : bar view or chart view
view_style_4 = st.radio("Choose your view",('Bar chart view', 'pie chart view'))
if view_style_4 == 'Bar chart view':
     st.write(most_listen_type_music)
else:
	st.write(pie_fig_type_musique)

#GRAPH 10 : Music feature analysis
st.subheader('Musique feature analysis')
spotify_audio_feature_mean = audio_features_mean(df_library_spotify_genre_mood,10)
col1, col2, col3,col4,col5,col6,col7 = st.columns(7)
col1.metric("Energy mean",round(spotify_audio_feature_mean.loc[0,"mean"],2))
col2.metric("Loudness mean",round(spotify_audio_feature_mean.loc[1,"mean"],2))
col3.metric("Speechness mean",round( spotify_audio_feature_mean.loc[2,"mean"],2))
col4.metric("Valence mean",round(spotify_audio_feature_mean.loc[3,"mean"],2))
col5.metric("Liveness mean",round(spotify_audio_feature_mean.loc[4,"mean"],2))
col6.metric("Tempo mean", round(spotify_audio_feature_mean.loc[5,"mean"],1))
col7.metric("danceability mean",round(spotify_audio_feature_mean.loc[6,"mean"],2))

st.subheader('Audio Feature definiiton')
st.markdown('**Energy** : Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy.')
st.markdown('**Loudness** : The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track. Values typical range between -60 and 0 db.')
st.markdown('**Speechness** : Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value.')
st.markdown('**Valence** : A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).')
st.markdown('**Liveness** : Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live.')
st.markdown('**Tempo** : The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.')
st.markdown('**danceability** : Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.')




#GRAPH 11: music variation analysis
st.subheader('Is my music have an high variation ?')

spotify_audio_feature_std = audio_features_std(df_library_spotify_genre_mood,10)
#merge audio feature mean df and audio feature std df into a single dataframe
audio_feature_mean_std = spotify_audio_feature_std.merge(spotify_audio_feature_mean,on="feature", how="left")
#Got the same data frame without tempo and loudness feature, because their scaling is diferent
audio_feature_mean_std_without_tempo_loudness = audio_feature_mean_std.drop([1,5])
#Got the same data frame with only tempo and loudness feature, because their scaling is diferent
audio_feature_mean_std_only_tempo_and_loudness = audio_feature_mean_std.drop([0,2,3,4,6])

#plot audio feature mean and std with two view
mean_ste_without_tempo_loudness = plot_mean_and_std_music_feature(audio_feature_mean_std_without_tempo_loudness,"feature","mean","standar_dev","Variety of musics")
mean_std_tempo_loudness = plot_mean_and_std_music_feature(audio_feature_mean_std_only_tempo_and_loudness,"feature","mean","standar_dev","Variety of musics with Tempo and Loudness")
#view audiio feature with or withoit tempo and loudness feature
view_style_5 = st.radio("Choose your view",('Audio feature witout tempo and loudness', 'audio feature tempo and loudness'))

if view_style_5 == 'Audio feature witout tempo and loudness':
     st.write(mean_ste_without_tempo_loudness)

else:
	st.write(mean_std_tempo_loudness)

#GRAPH 12 : Correlation between audio feature
st.subheader("Audio feature correlation analysis")
#Select only audio feature from the dataframe
df_library_spotify_genre_mood_feature = df_library_spotify_genre_mood[['energy',"loudness","speechiness","valence","liveness","tempo","danceability"]]
#Heat map
audio_feature_corr_heatmap = heat_map(df_library_spotify_genre_mood_feature)
#correlation between valence and danceability
scatter_valance_dansab = correlation_scatter(df_library_spotify_genre_mood,"valence","danceability","Correlation between Valence and Danceability")
cor_valence_danceability = stats.pearsonr(df_library_spotify_genre_mood['valence'], df_library_spotify_genre_mood['danceability'])[0]
#correlation between valence and loudness
scatter_valence_loudn = correlation_scatter(df_library_spotify_genre_mood,"valence","loudness","Correlation between Valence and Loudness")
cor_valence_loudness = stats.pearsonr(df_library_spotify_genre_mood['valence'], df_library_spotify_genre_mood['loudness'])[0]
#correlation between energy and loudness
scatter_energy_loudn = correlation_scatter(df_library_spotify_genre_mood,"energy","loudness","Correlation between Energy and Loudness")
cor_energy_loudness = stats.pearsonr(df_library_spotify_genre_mood['energy'], df_library_spotify_genre_mood['loudness'])[0]

# choose corelation view

view_style_6 = st.radio("Choose your view",('Correlation heat map', 'corr: valence/danceability','corr: valence/loudness','corr: energy/loudness'))

if view_style_6 == 'Correlation heat map':
	st.write(audio_feature_corr_heatmap)
elif view_style_6 == 'corr: valence/danceability':
	st.write(scatter_valance_dansab)
	st.metric("Pearson correlation",cor_valence_danceability)
elif view_style_6 == 'corr: valence/loudness':
	st.write(scatter_valence_loudn)
	st.metric("Pearson correlation",cor_valence_loudness)
else:
	st.write(scatter_energy_loudn)
	st.metric("Pearson correlation",cor_energy_loudness)

#GRAPH 13 : What kind of music am I listening
st.subheader('What kind of music am I analysing ?')
#got audio feature mean without tempo and loudness
audio_feature_mean_sans_tempo_loudness = spotify_audio_feature_mean.drop([1, 5])
#plot radar plot
polar_audio = plot_polar(audio_feature_mean_sans_tempo_loudness,'mean','feature',"Audio feature mean on Spotify playlist")
st.write(polar_audio)

#GRAPH 14 : Machine learning approach
st.subheader('Unsupervised Learning : Music clustering')

df_library_spotify_genre_mood_light = df_library_spotify_genre_mood[["artist_name","album_name","song_name","track_url","energy","loudness","speechiness","valence","liveness","tempo","danceability"]]
df_library_spotify_genre_mood_light = df_library_spotify_genre_mood.drop_duplicates(subset=["song_name"],keep='first')
spotify_audio_feature = df_library_spotify_genre_mood_light[["energy","loudness","speechiness","valence","liveness","tempo","danceability"]].values

#put all the feature in the same scale for the model
min_max_scaler = preprocessing.MinMaxScaler()
feature_scaled = min_max_scaler.fit_transform(spotify_audio_feature)

#There is too many feature, so we will reduse the dimension, lets find how many dimension can we reduce:
pca = PCA()
pca.fit(feature_scaled)
#pca.explained_variance_ratio_
plt.figure(figsize=(10,8))
plt.plot(range(1,8),pca.explained_variance_ratio_.cumsum(),marker='o')

#We can see thanks to the elbove methode that having 3 dimensions looks good
pca = PCA(n_components =3)
pca.fit(feature_scaled)
score_pca = pca.transform(feature_scaled)

#Now we scalled audio features and reduced audio dimensions, we will apply Kmean algo to see if we can cluster musics I've listened
kmeans_pca = cluster.KMeans(n_clusters=3,init="k-means++")
kmeans_pca.fit(score_pca)
# Regrouping all the results
df_library_spotify_genre_mood_light_pcs_kmean = pd.concat([df_library_spotify_genre_mood_light.reset_index(drop=True),pd.DataFrame(score_pca)],axis=1)
df_library_spotify_genre_mood_light_pcs_kmean.columns.values[-3:]=["Component_1","Component_2","Component_3"]
df_library_spotify_genre_mood_light_pcs_kmean["Segment-K-mean"] = kmeans_pca.labels_

kmean_cluster = px.scatter_3d(df_library_spotify_genre_mood_light_pcs_kmean, x='Component_1', y='Component_2', z='Component_3',color='Segment-K-mean')
st.write(kmean_cluster)

#GRAPH 15 : Number of song per cluster
st.subheader('Number of music per Cluster')
#find the number of song per cluster
number_song_cluster = number_song_per_cluster(df_library_spotify_genre_mood_light_pcs_kmean,'Segment-K-mean',"song_name")
#plot the viex with a bar chart
Number_of_cluster = px.bar(number_song_cluster, x='Segment-K-mean', y='nbre_chanson_par cluster', color='Segment-K-mean')
st.write(Number_of_cluster)

#GRAPH 16 : Audio feature analysis per cluster
st.subheader('Audio feature analysis per cluster')
audio_feature_per_cluster, ax = plt.subplots(figsize=(15,7))
df_library_spotify_genre_mood_light_pcs_kmean.groupby('Segment-K-mean').agg({'energy':'mean', 
                         'speechiness':'mean','valence': 'mean','liveness':'mean','danceability':'mean'}).plot.bar(ax=ax)

audio_feature_tempo_and_loudness, ax = plt.subplots(figsize=(15,7))
df_library_spotify_genre_mood_light_pcs_kmean.groupby('Segment-K-mean').agg({'tempo':'mean', 
                         'loudness':'mean'}).plot.bar(ax=ax)

view_style_6 = st.radio("Choose your view",('audio_feature_per_cluster', 'audio_feature_tempo_and_loudness'))

if view_style_6 == 'audio_feature_per_cluster':
	st.write(audio_feature_per_cluster)
else:
	st.write(audio_feature_tempo_and_loudness)

#GRAPH 17 : Music per cluster
#cluster 0
st.subheader('cluster 0')
df_library_spotify_genre_mood_light_pcs_kmean_cluster_0 = df_library_spotify_genre_mood_light_pcs_kmean[df_library_spotify_genre_mood_light_pcs_kmean["Segment-K-mean"]==0]
st.write(df_library_spotify_genre_mood_light_pcs_kmean_cluster_0["song_name"])

#cluster 1
st.subheader('cluster 1')
df_library_spotify_genre_mood_light_pcs_kmean_cluster_1 = df_library_spotify_genre_mood_light_pcs_kmean[df_library_spotify_genre_mood_light_pcs_kmean["Segment-K-mean"]==1]
st.write(df_library_spotify_genre_mood_light_pcs_kmean_cluster_1["song_name"])

#Cluster 2
st.subheader('cluster 2')
df_library_spotify_genre_mood_light_pcs_kmean_cluster_2 = df_library_spotify_genre_mood_light_pcs_kmean[df_library_spotify_genre_mood_light_pcs_kmean["Segment-K-mean"]==2]
st.write(df_library_spotify_genre_mood_light_pcs_kmean_cluster_2["song_name"])








