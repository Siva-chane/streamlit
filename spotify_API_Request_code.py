import requests
import pandas as pd
import json

#1)GET MUSIC TYPE

#Spliting the dataframe to fit with the maximum request of Spotify API
df_library_spotify_light = df_library_spotify.head(50)
df_library_spotify_50_100 = df_library_spotify.iloc[50:100]
df_library_spotify_100_150 = df_library_spotify.iloc[100:150]
df_library_spotify_150 = df_library_spotify.iloc[150:]

# API token
CLIENT_ID = 'XXXX'
CLIENT_SECRET = 'XXXX'

# authentication URL
AUTH_URL = 'https://accounts.spotify.com/api/token'

# POST
auth_response = requests.post(AUTH_URL, {
    'grant_type': 'client_credentials',
    'client_id': CLIENT_ID,
    'client_secret': CLIENT_SECRET,
})

# convert the response to JSON
auth_response_data = auth_response.json()

# save the access token
access_token = auth_response_data['access_token']
headers = {'Authorization': 'Bearer {token}'.format(token=access_token)}
BASE_URL = 'https://api.spotify.com/v1/'

# create blank dictionary to store track URI, artist URI, and genres
dict_genre = {}

# convert track_uri column to an iterable list
track_uris = df_library_spotify_150['track_url'].to_list()

# loop through track URIs and pull artist URI using the API,
# then use artist URI to pull genres associated with that artist
# store all these in a dictionary
for t_uri in track_uris:
    
    dict_genre[t_uri] = {'artist_uri': "", "genres":[]}
    
    r = requests.get(BASE_URL + 'tracks/' + t_uri, headers=headers)
    r = r.json()
    a_uri = r['artists'][0]['uri'].split(':')[2]
    dict_genre[t_uri]['artist_uri'] = a_uri
    
    s = requests.get(BASE_URL + 'artists/' + a_uri, headers=headers)
    s = s.json()
    dict_genre[t_uri]['genres'] = s['genres']

df_genre = pd.DataFrame.from_dict(dict_genre, orient='index')
df_genre.insert(0, 'track_uri', df_genre.index)
df_genre.reset_index(inplace=True, drop=True)
df_genre_expanded = df_genre.explode('genres')

#Concatenation of results
df_genre_expanded_50 = df_genre_expanded.copy()
df_genre_expanded_50_A_100 = df_genre_expanded.copy()
df_genre_expanded_100_A_150 = df_genre_expanded.copy()
df_genre_expanded_150 = df_genre_expanded.copy()

df_full_genre = pd.concat([df_genre_expanded_50,df_genre_expanded_50_A_100,df_genre_expanded_100_A_150,df_genre_expanded_150])
df_full_genre.reset_index(inplace=True)

#Merge with the initial YourLibrary dataset to get all the information in one place
df_library_spotify_genre = df_library_spotify.merge(df_full_genre,left_on="track_url",right_on="track_uri",how="left")
df_library_spotify_genre.drop(["track_uri"],axis=1,inplace=True)


#2) GET AUDIO FEATURE
#To get Audio feature we need to use URI columns, because the df_library_spotify_genre is an explode dataframe, there is duplicates on URL request
#We drop duplicates to get unique URI, then we are not making useless request

df_library_spotify_genre_unique = df_library_spotify_genre.drop_duplicates(subset=["URI"],keep="first")
df_library_spotify_genre_unique


client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
sp.trace=False

#your uri list goes here
df_library_spotify_genre
s_list = list(df_library_spotify_genre_unique["URI"])
#s_list = ['spotify:track:2d7LPtieXdIYzf7yHPooWd','spotify:track:0y4TKcc7p2H6P0GJlt01EI','spotify:track:6q4c1vPRZREh7nw3wG7Ixz','spotify:track:54KFQB6N4pn926IUUYZGzK','spotify:track:0NeJjNlprGfZpeX2LQuN6c']

#put uri to dataframe
df = pd.DataFrame(s_list)
df.columns = ['URI']

df['energy'] = ''*df.shape[0]
df['loudness'] = ''*df.shape[0]
df['speechiness'] = ''*df.shape[0]
df['valence'] = ''*df.shape[0]
df['liveness'] = ''*df.shape[0]
df['tempo'] = ''*df.shape[0]
df['danceability'] = ''*df.shape[0]

for i in range(0,df.shape[0]):
    time.sleep(random.uniform(3, 6))
    URI = df.URI[i]
    features = sp.audio_features(URI)
    df.loc[i,'energy'] = features[0]['energy']
    df.loc[i,'speechiness'] = features[0]['speechiness']
    df.loc[i,'liveness'] = features[0]['liveness']
    df.loc[i,'loudness'] = features[0]['loudness']
    df.loc[i,'danceability'] = features[0]['danceability']
    df.loc[i,'tempo'] = features[0]['tempo']
    df.loc[i,'valence'] = features[0]['valence']
    uri=0

#Collecting result
#We merge the results with the previous dataframe in order to have all the information in one place 
df_library_spotify_genre_mood = df_library_spotify_genre.merge(df, on="URI",how='left')
#I'm exporting this file on CSV in order to analyse the dataframe without executing the whole code
df_library_spotify_genre_mood.to_csv("df_library_spotify_genre_mood.csv")

