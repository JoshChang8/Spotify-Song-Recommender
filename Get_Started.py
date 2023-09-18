import streamlit as st
import os

port = int(os.environ.get('PORT', 8501))

st.set_page_config(
    page_title="Get Started",
    page_icon="🎵", 
)

st.write("# Spotify Recommender System 🎵")

st.sidebar.info("Select a feature above.")

st.success("#####  Share your playlist link to uncover in-depth playlist analytics with curated song recommendations to enhance your playlist and elevate your music experience!")
st.write("Annually, Spotify releases its popular feature, '*Spotify Wrapped*,' providing users with a personalized summary of their listening habits. Over time, it has gained immense popularity, with Spotify users eagerly awaiting each year to proudly showcase their favorite genres, top artists, and most-played songs. However, this feature occurs annually and encompasses the user's overall listening history.")
st.write("The Spotify Song Recommender System allows users to input any playlist, conducting an in-depth analysis and generating a personalized song list. This curation is guided by the audio features of each track within the playlist, utilizing mathematical similarity measures to ensure a tailored music experience for each playlist.")
st.divider()
st.write('##### To read my detailed Medium article that describes my process of completing this project, visit the link below:')
st.write('#### [Data-Driven Music Exploration: Building a Spotify Song Recommender](https://medium.com/@joshjc038/data-driven-music-exploration-building-a-spotify-song-recommender-5780cabfe194)')
st.divider()
st.write('### Video Demonstration of Spotify Recommender System')
st.video('https://www.youtube.com/watch?v=MjjKtIvT-EA')
st.divider()
st.write("## Playlist Analysis 📈")
st.write("Getting bored of the same old playlist and want to discover new tracks that 'fit the vibe'? Share your playlist link to unveil a comprehensive analysis of your playlist's audio features, genre distribution, and most popular songs. Leveraging the audio attributes and genres of the songs in your playlist, mathematical similarity measures are used to recommend 30 new tracks to expand your musical horizons. Plus, you'll enjoy a 30-second audio preview of each recommended song!")
st.write("## Musical Middle Ground ✨")
st.write("Have you ever found yourself in a car with a group of friends arguing over who gets control of the aux cord? Imagine being able to craft the perfect playlist that caters to everyone's unique musical tastes. That's precisely what the Musical Middle Ground feature offers! Discover shared songs, explore playlist feature metrics, and observe genre distributions that address everyone's preferences. By using track feature metrics from songs across all the playlists, we create a curated list of songs that seamlessly blend the diverse tastes of multiple playlists!")
