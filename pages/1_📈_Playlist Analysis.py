# Allows user to input link to playlist and analytics regarding playlist will be given along with a recommendation of songs

# TODO
# - show progress bar for calculating results... in sidebar

import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import spotipy
import json
from spotipy.oauth2 import SpotifyOAuth
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
import multiprocessing

#connect to spotify API
# Set your Spotify API credentials
client_id = '14c84923f9ac478abf582c59dcc6f59c'
client_secret = '8dfc8d50a3164779bdb8d74010f913b5'
redirect_uri = 'http://localhost:3000'

# Initialize the Spotipy client with authentication
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id, client_secret, redirect_uri))

# Parallelization process artist names and retrieve genres
def process_artist(artist_name, sp):
    search_results = sp.search(q=artist_name, type='artist')
    genres_info = []

    if 'artists' in search_results and 'items' in search_results['artists']:
        artists = search_results['artists']['items']

        for artist in artists:
            if artist['name'] == artist_name:
                genres_info = artist.get('genres', [])
                break

    genre_string = ', '.join(genres_info) if genres_info else 'No Genre Found'
    return genre_string

# Multiprocessing to process artist names and retrive genres
def fetch_artist_genres(artist_name):
    search_results = sp.search(q=artist_name, type='artist')
    genres_info = []

    if 'artists' in search_results and 'items' in search_results['artists']:
        artists = search_results['artists']['items']

        for artist in artists:
            if artist['name'] == artist_name:
                genres_info = artist.get('genres', [])
                break

    genre = ', '.join(genres_info) if genres_info else 'No Genre Found'
    return genre

# Read csv file
@st.cache_data
def read_csv():
    return pd.read_csv('spotify_songs.csv')

# Change spotify playlist dictionary of links to URIs
def get_uri(playlist):
    playlist = playlist[34:56]
   
    try:
        playlist_tracks = sp.playlist_tracks(playlist)
    
    except spotipy.SpotifyException as e:
        if "Unsupported URL / URI" in str(e):
            st.error('This playlist link is invalid', icon='🚨')
    
    return playlist

# Using playlist URI, get df of spotify songs in playlist
def get_df(uri, playlist_size):
    playlist_tracks = sp.playlist_tracks(uri, limit=playlist_size)

    # Create lists to hold track titles and artist names
    titles, artists, uri = [], [], []

    st.toast('Gathering Playlist Data', icon='🔎')

    # Iterate through the tracks and collect data
    for item in playlist_tracks['items']:
        track = item['track']
        titles.append(track['name'])
        artist_names = ', '.join([artist['name'] for artist in track['artists']])
        artists.append(artist_names)
        uri.append(track['uri'])

    # Create a DataFrame
    data = {'Title': titles, 'Artist': artists, 'uri': uri}
    playlist = pd.DataFrame(data)

    # create new feature columns and assign null values 
    new_feat = ['danceability', 'energy', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    for item in new_feat:
        playlist[item] = 0

    # fill null with feature values 
    for i in range(len(playlist)):
        track_uri = playlist.iloc[i].uri
        audio_features = sp.audio_features(track_uri)
        json_string = json.dumps(audio_features[0])
        dictionary = json.loads(json_string)
        
        #update feature values
        for feature in new_feat:
            playlist.loc[i, feature] = dictionary[feature]
    
    # Get Genre, Popularity, and Year
    # Get track URIs
    track_uris = playlist['uri'].tolist()

    # Fetch track information for all tracks in the playlist
    track_infos = [sp.track(uri) for uri in track_uris]

    # Extract release years and popularity using list comprehensions
    release_years = [int(info['album']['release_date'].split('-')[0]) for info in track_infos]
    popularity_list = [info['popularity'] for info in track_infos]

    # Create a list of artist names from the playlist
    artist_names = playlist['Artist'].tolist()

    # Create an empty list to store genres
    genres = []

    st.toast('Generating Playlist Details', icon='⚙️')

    # for artist_name in artist_names:
    #     search_results = sp.search(q=artist_name, type='artist')
    #     genres_info = []

    #     if 'artists' in search_results and 'items' in search_results['artists']:
    #         artists = search_results['artists']['items']

    #         for artist in artists:
    #             if artist['name'] == artist_name:
    #                 genres_info = artist.get('genres', [])
    #                 break

    #     genre = ', '.join(genres_info) if genres_info else 'No Genre Found'
    #     genres.append(genre)
      
    # Fill genre for each song using Parallelization
    genres = Parallel(n_jobs=1)(delayed(process_artist)(artist_name, sp) for artist_name in artist_names)

    # multiprocessing
    # if __name__ == "__main__":
    #     # Create a multiprocessing pool
    #     with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #         # Use the pool.map method to parallelize the execution
    #         genres = pool.map(fetch_artist_genres, [(artist_name) for artist_name in artist_names])


    # Add the collected data to the DataFrame
    playlist['Year'] = release_years
    playlist['Popularity'] = popularity_list
    playlist['Genre'] = genres

    return playlist     

#Show genre breakdown
def show_genre_breakdown(playlist):
    genre_count = {}
    genres = ['acoustic', 'alt-rock', 'ambient', 'blues', 'chill', 'classical', 'club', 'country', 'dance', 'dancehall', 'disco', 'dub', 'edm', 'electro', 'emo', 'folk', 'funk', 'gospel', 'goth', 'groove', 'guitar', 'hip-hop', 'house', 'indie', 'jazz', 'k-pop', 'metal', 'opera', 'party', 'piano', 'pop', 'punk', 'rock', 'rock-n-roll', 'romance', 'sad', 'salsa', 'samba', 'singer-songwriter', 'sleep', 'songwriter', 'soul', 'spanish', 'tango', 'techno']
    # using the genre, find substrings of genres and assign values of 1 if found, create dict with count of genres
    for genre in genres:
        playlist['genre_'+genre] = playlist['Genre'].str.contains(genre).astype(int)
        if playlist['genre_'+genre].sum() > 0:
            genre_count[genre] = playlist['genre_'+genre].sum()
    
    st.write('### Genre Distribution')
    # Create a pie chart using Plotly
    labels = list(genre_count.keys())
    values = list(genre_count.values())
    trace = go.Pie(labels=labels, values=values)
    genre_pie_chart = go.Figure(data=[trace])
    st.plotly_chart(genre_pie_chart)

    #get top 3 genres for recommendation 
    top_3 = sorted(genre_count, key=genre_count.get, reverse=True)[:3]
    return top_3
  
# Show most popular songs in playlist
def show_most_popular_songs(playlist):
    st.write('### Popular Songs')
    popularity_playlist = playlist.sort_values(by='Popularity', ascending=False)
    columns_shown = ['Title', 'Artist', 'Popularity']

    if len(popularity_playlist) > 5:
        top_five = popularity_playlist.head(5)
        top_five = top_five[columns_shown].reset_index(drop=True)
        st.dataframe(top_five)
    else:
        st.write("There are not enough songs in this playlist.")

# Show Track Title, Artist, Genre, Year, Popularity
def display_df(playlist):
    # Change to type of Year column from int to string
    playlist['Year'] = playlist['Year'].astype(str)

    # Select and return the desired columns
    display_features = ['Title', 'Artist', 'Genre', 'Year', 'Popularity']
    result_df = playlist[display_features]

    st.dataframe(result_df)

# Get song recommendations based on user's playlist
def get_recommendations(playlist, top_3_genres):
    feat_vec = read_csv()

    st.toast('Computing Song Recommendations', icon='🎵')

    # need to have the same shape and order of columns in order to run cosine similarity
    # make buckets based on year range to match feature vector
    playlist['year_2000-2004'] = playlist['Year'].astype(int).apply(lambda year: 1 if year>=2000 and year<2005 else 0)
    playlist['year_2005-2009'] = playlist['Year'].astype(int).apply(lambda year: 1 if year>=2005 and year<2010 else 0)
    playlist['year_2010-2014'] = playlist['Year'].astype(int).apply(lambda year: 1 if year>=2010 and year<2015 else 0)
    playlist['year_2015-2019'] = playlist['Year'].astype(int).apply(lambda year: 1 if year>=2015 and year<2020 else 0)
    playlist['year_2020-2023'] = playlist['Year'].astype(int).apply(lambda year: 1 if year>=2020 and year<2024 else 0)
    playlist = playlist.drop(columns=['Year'])

    #sort alphabetical order to match columns with feat_vec df
    playlist['popularity'] = playlist['Popularity']
    playlist = playlist.sort_index(axis=1)
    columns_to_drop = ['Artist', 'Genre', 'Title', 'Popularity']
    playlist = playlist.drop(columns=columns_to_drop)

    #change scale to match scale of feature vector playlist 
    # popularity scale: 1-100, loudness scale: -60-0, tempo scale: 0-250, scale features from 0-1 
    #add min and max values for each row to establish min and max values, then once scaling is done, remove min and max columns
    min_row = {'popularity': '0', 'loudness': '-60', 'tempo': '0'}
    max_row = {'popularity': '100', 'loudness': '0', 'tempo': '250'}

    playlist = playlist.append(min_row, ignore_index=True)
    playlist = playlist.append(max_row, ignore_index=True)

    # scale popularity, loudness, and tempo features to 0-1
    scale = ['popularity', 'loudness', 'tempo']
    scaler = MinMaxScaler()
    playlist[scale] = scaler.fit_transform(playlist[scale])

    # drop min and max values
    playlist = playlist.iloc[:-2]
    
    #create column avgs as a parameter for cosine similarity
    column_avg = playlist.mean()
    playlist_cosine_sim = pd.DataFrame([column_avg], index=['Average'])

    #make sure feature vector playlist is same shape for cosine similarity
    feat_vec_cosine_sim = feat_vec.drop('track_id', axis=1)

    # generate similarity scores
    similarity_scores = cosine_similarity(feat_vec_cosine_sim, playlist_cosine_sim)
    feat_vec['similarity_score'] = similarity_scores

    #sort df from highest to lowest by similarity score and show reccomendations
    top_similarities = feat_vec.sort_values(by='similarity_score', ascending=False)

    #remove rows in recommendations from top_similarities where IDs match with playlist IDs
    top_similarities = top_similarities[~top_similarities['track_id'].isin(playlist['uri'])]

    if len(top_3_genres) == 3:
        # get song recs from top 3 genres, will have to remove some coolumns after if they are ethnic songs
        first_genre = top_similarities.loc[top_similarities['genre_'+top_3_genres[0]] == 1].head(45)
        second_genre = top_similarities.loc[top_similarities['genre_'+top_3_genres[1]] == 1].head(30)
        third_genre = top_similarities.loc[top_similarities['genre_'+top_3_genres[2]] == 1].head(15)

        top_similarities = pd.concat([first_genre, second_genre, third_genre], ignore_index=True)

    else:
        top_similarities = top_similarities.head(90)

    top_similarities['track'] = [None]*len(top_similarities)
    top_similarities['artist'] = [None]*len(top_similarities)
    top_similarities['preview'] = [None]*len(top_similarities)
    
    # get track name, artist, and 30s audio clip url
    for i in range(len(top_similarities)):
        track_info = sp.track(top_similarities.iloc[i,55])
        track_name = track_info['name']
        artist_name = track_info['artists'][0]['name']
        preview_url = track_info['preview_url']
    
        top_similarities.iloc[i, 63] = track_name
        top_similarities.iloc[i, 64] = artist_name
        top_similarities.iloc[i, 65] = preview_url

    # Get genres of each track in playlist
    # Create a list of artist names from the playlist
    artist_names = top_similarities['artist'].tolist()
    
    # Create an empty list to store genres
    genres = []

    st.toast('Displaying Song Recommendations', icon='✨')

    # Fill genre for each song using Parallelization
    genres = Parallel(n_jobs=1)(delayed(process_artist)(artist_name, sp) for artist_name in artist_names)

    # multiprocessing
    # if __name__ == "__main__":
    #     # Create a multiprocessing pool
    #     with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #         # Use the pool.map method to parallelize the execution
    #         genres = pool.map(fetch_artist_genres, [(artist_name) for artist_name in artist_names])

    # for artist_name in artist_names:
    #     search_results = sp.search(q=artist_name, type='artist')
    #     genres_info = []

    #     if 'artists' in search_results and 'items' in search_results['artists']:
    #         artists = search_results['artists']['items']

    #         for artist in artists:
    #             if artist['name'] == artist_name:
    #                 genres_info = artist.get('genres', [])
    #                 break

    #     genre = ', '.join(genres_info) if genres_info else 'No Genre Found'
    #     genres.append(genre)
   
    # Add the collected data to the DataFrame
    top_similarities['genre'] = genres

    # if songs in recs have any ethnic songs
    ethnic_genres = ['colombia', 'latin', 'mexican', 'puerto rican', 'dominican', 'italian', 'spanish', 'brasil', 'argentine', 'anime', 'japanese', 'indonesian', 'vietnamese', 'korean', 'chinese', 'taiwan', 'spanish']
    
    # remove any songs that have ethnic genres included
    mask = top_similarities['genre'].str.contains('|'.join(ethnic_genres), case=False)
    top_similarities.drop(top_similarities[mask].index, inplace=True)
    
    if len(top_3_genres) == 3:
        # 15 songs from 1st genre, 10 songs from 2nd genre, 5 songs from 3rd genre
        first_genre = top_similarities.loc[top_similarities['genre_'+top_3_genres[0]] == 1].head(15)
        second_genre = top_similarities.loc[top_similarities['genre_'+top_3_genres[1]] == 1].head(10)
        third_genre = top_similarities.loc[top_similarities['genre_'+top_3_genres[2]] == 1].head(5)
        top_similarities = pd.concat([first_genre, second_genre, third_genre], ignore_index=True)

    else:
        top_similarities = top_similarities.head(50)
 
    return top_similarities

# Show df of top 30 recs but in better format 
def format_recommendation_results(playlist_recs):
   
   #show only specific columns 
    display_features = ['track', 'artist', 'similarity_score', 'genre', 'preview']
    playlist_recs = playlist_recs[display_features]

    playlist_recs['similarity_score'] = (playlist_recs['similarity_score']*100).round(2)

    column_configuration = {
        "track": st.column_config.TextColumn("Track", help="Title of Track"),
        "artist": st.column_config.TextColumn("Artist", help="Artist of Track"),
        "similarity_score": st.column_config.NumberColumn("Similarity Score", help="Similarity score to the playlist"),
        "genre": st.column_config.TextColumn("Genres", help="Genres of Track"),
        "preview": st.column_config.LinkColumn(
            "30s Audio Preview", help="30s preview clip of track"
        ),

    }
    st.data_editor(playlist_recs, column_config=column_configuration, hide_index=True, num_rows='fixed', disabled=True)

st.set_page_config(page_title="Playlist Analysis", page_icon="📈")

st.markdown("# Playlist Analysis 📈")
st.write("Share your playlist link to unveil a comprehensive analysis of your playlist's audio features, genre distribution, and most popular songs. Leveraging the audio attributes and genres of the songs in your playlist, mathematical similarity measures are used to recommend 30 new tracks to expand your musical horizons. Plus, you'll enjoy a 30-second audio preview of each recommended song!")

st.sidebar.header("Playlist Parameters")

playlist = ''

with st.sidebar.form(key='Form1'):
    # User inputs name and link to playlists
    playlist = st.text_input('Enter Playlist Link') 
    playlist_size = st.slider('Size of playlist', 15, 100, 50, 5)
    submitted_playlist = st.form_submit_button(label = 'Find Playlist 🔎')

if playlist == '':
    st.warning('Please input a valid playlist link to analyze.')
    st.write('To make sure the program can read your playlist, please input the playlist link in this format:')
    st.text('https://open.spotify.com/playlist/...')
    st.write('Playlist links in this format can be found through Spotify web and desktop app, NOT the mobile app')
    st.write('Provided is a sample link to run a playlist analysis: https://open.spotify.com/playlist/37i9dQZF1DX5Q5wA1hY6bS?si=9d42312b259e4c60')


else:
    playlist = get_uri(playlist)

    # Print playlist name
    playlist_info = sp.playlist(playlist)
    playlist_name = playlist_info['name']
    st.write('# '+playlist_info['name']+' Playlist Analytics')

    playlist = get_df(playlist, playlist_size)

    # Get values of Energy, Danceability, and Valence
    st.write('### Feature Playlist Ratings')
    playlist_avg = playlist.mean()
    col1, col2, col3 = st.columns(3)
    col1.metric("Energy", int(playlist_avg['energy']*100))
    col2.metric("Danceability", int(playlist_avg['danceability']*100))
    col3.metric("Valence", int(playlist_avg['valence']*100))
    # Dropdown for info about features
    with st.expander("Feature Description"):
        st.write('Energy - represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy. A value of 0 is the least energetic and 100 is most energetic')
        st.write('Danceability - describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall. A value of 0 is least danceable and 100 is most danceable')
        st.write('Valence - positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry). A value of 0 is the lowest valence and 100 has the highest valence')
        st.write('Additional Audio Feature Descriptions: [Song Audio Features](https://developer.spotify.com/documentation/web-api/reference/get-audio-features)')
    #st.divider()

    # Show breakdown of genres 
    top_3_genres = show_genre_breakdown(playlist)

    # Show top 5 most popular songs
    show_most_popular_songs(playlist)
    #st.divider()

    # Show dataframe
    with st.expander("Checkout Playlist Songs"):
        display_df(playlist)

    # Song Recommendations
    st.write('### Song Recommendations')
    #st.dataframe(playlist)
    playlist = get_recommendations(playlist, top_3_genres)

    format_recommendation_results(playlist)

# tests
# Serendipity: https://open.spotify.com/playlist/51mwSPAk0bqVFM4Lz0IXZ1?si=f6f564a6cc564c89
# Wild & Free: https://open.spotify.com/playlist/37i9dQZF1DX5Q5wA1hY6bS?si=9d42312b259e4c60
# Soft Pop Hits: https://open.spotify.com/playlist/37i9dQZF1DWTwnEm1IYyoj?si=93c3dd7ae383404b 








