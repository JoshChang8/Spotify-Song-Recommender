# Allows a maximum of 3 playlist and drawing analytics from each song in all playlists, songs will be recommended

import streamlit as st
import pandas as pd
import spotipy
import json
import plotly.graph_objs as go
from spotipy.oauth2 import SpotifyOAuth
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed 
import multiprocessing

# TODO
# if time permits, create weightings for certain songs if they are combined

#connect to spotify API
# Set your Spotify API credentials
client_id = 'CLIENT_ID'
client_secret = 'CLIENT_SECRET'
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
def get_uri(playlists):
    for key in playlists:
        playlists[key] = playlists[key][34:56]
    return playlists

# Using playlist URI, get df of spotify songs in playlist
def get_df(uri):
    playlist_tracks = sp.playlist_tracks(uri, limit=50)

    # Create lists to hold track titles and artist names
    titles, artists, uri = [], [], []

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
    
    return playlist     

# Show Track Title, Artist, Genre, Year, Popularity
def display_df(playlist):
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

    # Change to type of Year column from int to string
    playlist['Year'] = playlist['Year'].astype(str)

    # Select and return the desired columns
    display_features = ['Title', 'Artist', 'Genre', 'Year', 'Popularity']
    result_df = playlist[display_features]

    return result_df

def show_user_df(playlist):
    # Select and return the desired columns
    display_features = ['Title', 'Artist', 'Genre', 'Year', 'Popularity']
    result_df = playlist[display_features]
    st.dataframe(result_df, hide_index=True)

# Combine playlists, use overloading to take in 2 or 3 parameters 
def combine_playlists(*args, **kwargs):
    if len(args) == 2:
        combined = pd.concat([args[0], args[1]], ignore_index=True)
    if len(args) == 3:
        combined = pd.concat([args[0], args[1], args[2]], ignore_index=True)
    
    duplicates = combined['uri'].duplicated()
    duplicate_rows = combined[duplicates]
    combined = combined.drop_duplicates()
    st.write('## Middleground Playlist')
    st.write('A combination of all songs into one playlist.')
    show_user_df(combined)
    if combined.shape[0] >= 1:
        st.write('## Tracks in Common')
        show_user_df(duplicate_rows)
    return combined

# Show metrics of 
def get_metrics(df):
    st.write('## Middleground Playlist Feature Metrics')
    
    averages = df.mean()

    energy = averages['energy']
    danceability = averages['danceability']
    valence = averages['valence']

    col1, col2, col3 = st.columns(3)
    col1.metric('Energy', int(energy*100))
    col2.metric('Danceability', int(danceability*100))
    col3.metric('Valence', int(valence*100))

    # Dropdown for info about features
    with st.expander("Feature Description"):
        st.write('Energy - represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy. A value of 0 is the least energetic and 100 is most energetic')
        st.write('Danceability - describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall. A value of 0 is least danceable and 100 is most danceable')
        st.write('Valence - positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry). A value of 0 is the lowest valence and 100 has the highest valence')
        st.write('Additional Audio Feature Descriptions: [Song Audio Features](https://developer.spotify.com/documentation/web-api/reference/get-audio-features)')

#Show distribution of genres in playlist
def show_genre_breakdown(playlist):
    genre_count = {}
    genres = ['acoustic', 'alt-rock', 'ambient', 'blues', 'chill', 'classical', 'club', 'country', 'dance', 'dancehall', 'disco', 'dub', 'edm', 'electro', 'emo', 'folk', 'funk', 'gospel', 'goth', 'groove', 'guitar', 'hip-hop', 'house', 'indie', 'jazz', 'k-pop', 'metal', 'opera', 'party', 'piano', 'pop', 'punk', 'rock', 'rock-n-roll', 'romance', 'sad', 'salsa', 'samba', 'singer-songwriter', 'sleep', 'songwriter', 'soul', 'spanish', 'tango', 'techno']
    # using the genre, find substrings of genres and assign values of 1 if found, create dict with count of genres
    for genre in genres:
        playlist['genre_'+genre] = playlist['Genre'].str.contains(genre).astype(int)
        if playlist['genre_'+genre].sum() > 0:
            genre_count[genre] = playlist['genre_'+genre].sum()

    st.write('## Genre Distribution')
    # Create a pie chart using Plotly
    labels = list(genre_count.keys())
    values = list(genre_count.values())
    trace = go.Pie(labels=labels, values=values)
    genre_pie_chart = go.Figure(data=[trace])
    st.plotly_chart(genre_pie_chart)

    #get top 3 genres for recommendation 
    top_3 = sorted(genre_count, key=genre_count.get, reverse=True)[:3]
    return top_3

# Show metrics of 
def get_recommendations(playlist, top_3_genres):
    feat_vec = read_csv()

    st.toast('Computing Song Recommendations', icon='🎵')

    # need to have the same shape and order of columns in order to run cosinesimilarity
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
    st.write('## Song Recommendations')
    st.data_editor(playlist_recs, column_config=column_configuration, hide_index=True, num_rows='fixed', disabled=True)

st.set_page_config(page_title="Recommendations", page_icon="✨")

st.markdown("# Musical Middle Ground ✨")
st.write("Discover shared songs, explore playlist feature metrics, and observe genre distributions that address everyone's preferences. By using track feature metrics from songs across all the playlists, we create a curated list of songs that seamlessly blend the diverse tastes of multiple playlists!")

playlist_names = []
playlist_links = []

st.sidebar.header("Playlist Import Parameters")
num_playlists = st.sidebar.number_input('Number of Playlists', min_value=2, max_value=3, value=2, step=1)

with st.sidebar.form(key='Form1'):
    # User inputs name and link to playlists
    for i in range(num_playlists):
        playlist_names.append(st.text_input('Enter Name of Playlist '+str(i+1), key=i+3)) 
        playlist_links.append(st.text_input('Enter Link of Playlist '+str(i+1), key=i)) 

    #convert playlist names and links to dictionary
    playlists = {key:value for key, value in zip(playlist_names, playlist_links)}
    playlists = get_uri(playlists)
    
    submitted_playlist = st.form_submit_button(label = 'Find Playlists 🔎')

if submitted_playlist:
    playlist_keys = list(playlists.keys())
    st.toast('Searching Spotify for Playlists', icon='🔎')

    st.write('## Individual Playlists')
    with st.expander('See information for each song in playlist'):
        if len(playlists) == 2:
            tab1, tab2 = st.tabs(list(playlists.keys()))
            # Playlist 1 Dataframe
            playlist_1 = get_df(playlists[playlist_keys[0]])
            display_playlist_1 = display_df(playlist_1)
            tab1.dataframe(display_playlist_1, hide_index=True)
            st.toast('Generated Data from '+playlist_keys[0]+' Playlist', icon='🎵')
            # Playlist 2 Dataframe
            playlist_2 = get_df(playlists[playlist_keys[1]])
            display_playlist_2 = display_df(playlist_2)
            tab2.dataframe(display_playlist_2, hide_index=True)
            st.toast('Generated Data from '+playlist_keys[1]+' Playlist', icon='🎵')
        
        else:
            tab1, tab2, tab3 = st.tabs(list(playlists.keys()))
            # Playlist 1 Dataframe
            playlist_1 = get_df(playlists[playlist_keys[0]])
            display_playlist_1 = display_df(playlist_1)
            tab1.dataframe(display_playlist_1)
            st.toast('Generated Data from '+playlist_keys[0]+' Playlist', icon='🎵')
            # Playlist 2 Dataframe
            playlist_2 = get_df(playlists[playlist_keys[1]])
            display_playlist_2 = display_df(playlist_2)
            tab2.dataframe(display_playlist_2)
            st.toast('Generated Data from '+playlist_keys[1]+' Playlist', icon='🎵')
            #Display Playlist Dataframe
            playlist_3 = get_df(playlists[playlist_keys[2]])
            display_playlist_3 = display_df(playlist_3)
            tab3.dataframe(display_playlist_3)
            st.toast('Generated Data from '+playlist_keys[2]+' Playlist', icon='🎵')

    if len(playlists) == 2:
        combined = combine_playlists(playlist_1, playlist_2)
    else:
        combined = combine_playlists(playlist_1, playlist_2, playlist_3)

    get_metrics(combined)
    st.toast('Generating Genre Breakdown', icon='🥧')
    top_3_genres = show_genre_breakdown(combined)
    format_recommendation_results(get_recommendations(combined, top_3_genres))

else:
    st.warning('Please input playlist paramters')
    st.write('To make sure the program can read your playlist, please input the playlist link in this format:')
    st.text('https://open.spotify.com/playlist/...')
    st.write('Playlist links in this format can be found through Spotify web and desktop app, NOT the mobile app')
    st.write('Provided is a sample link of a playlist: https://open.spotify.com/playlist/37i9dQZF1DX5Q5wA1hY6bS?si=9d42312b259e4c60')


# tests
# Serendipity: https://open.spotify.com/playlist/51mwSPAk0bqVFM4Lz0IXZ1?si=f6f564a6cc564c89
# Wild & Free: https://open.spotify.com/playlist/37i9dQZF1DX5Q5wA1hY6bS?si=9d42312b259e4c60
# Soft Pop Hits: https://open.spotify.com/playlist/37i9dQZF1DWTwnEm1IYyoj?si=93c3dd7ae383404b 










