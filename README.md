# Spotify Song Recommender

> _A Spotify song recommendation system conducting analysis on a user’s playlist and delivering personalized recommendations._

## Read the detailed [Medium Article](https://medium.com/@joshjc038/data-driven-music-exploration-building-a-spotify-song-recommender-5780cabfe194) that describes my process of completing this project

https://github.com/JoshChang8/Spotify-Song-Recommender/assets/82460964/cb4a7230-0da7-4f16-a0e4-bdbbe011c307

## Spotify Song Recommendation System
This recommender system uses audio features and genres of each song within a playlist to curate a list of songs that closely align with the playlist’s characteristics. This project includes two distinct features: a playlist analytics tool and a musical middle ground feature.

### Playlist Analytics
The purpose of this feature was to mimic Spotify Wrapped but offer a more flexible and personalized experience. Unlike Spotify Wrapped which provides annual insights, the playlist analytics feature allows users to focus on a specific playlist of their choice. When the user enters a playlist link, they receive a comprehensive analytical playlist breakdown with tailored song recommendations.

### Musical Middle Ground
The Musical Middle Ground feature was developed to resolve arguments of who’s got aux. Users can input up to three playlist links and an analytical breakdown based on the playlists is provided along with tailored song recommendations that combine the musical preferences represented by the three playlists.

## Run this app on your local machine
1. Clone the GitHub Repository.
2. Download the required packages: ```pip install -r requirements.txt```
3. Create your own personal Spotify Developer account.
    - In order to use the Spotify API, Spotify requires you to make a developer account and create an app in order to get your Client ID and Client Secret. After this is done, you will need to request an access token using your client credentials. This access token gives you the permissions to access user data and resources (artist, tracks, albums). Make sure to fill in the Client ID and Secret values in the neccessary files. 
5. Start the Streamlit app: ```streamlit run Get_Started.py```




