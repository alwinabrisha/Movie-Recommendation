import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the movies data from CSV
movies_df = pd.read_csv('movies.csv')

# Display title
st.title("Movie Recommendation System")

# Check the first few rows to show movie titles
st.write("Available Movie Titles:")
st.write(movies_df['title'].head())  # Displaying the first 5 movie titles

# Function to recommend movies based on user input
def recommend_movies(movie_title):
    # Ensure the title matches the movie data (case-insensitive match)
    movie = movies_df[movies_df['title'].str.lower() == movie_title.lower()]

    if movie.empty:
        return "Movie not found! Try another title."

    # Extract features for recommendation (Here, using 'description' as the feature)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['description'].fillna(''))

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Find the index of the movie that matches the title
    idx = movie.index[0]

    # Get similarity scores for all movies with the given movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort movies based on similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top 5 similar movies
    sim_scores = sim_scores[1:6]

    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Recommend the top 5 movies
    recommended_movies = movies_df['title'].iloc[movie_indices]

    return recommended_movies

# Dropdown for movie selection
movie_title_input = st.selectbox("Select a movie title:", movies_df['title'])

# Display recommendations when a title is selected
if movie_title_input:
    recommendations = recommend_movies(movie_title_input)
    if isinstance(recommendations, str):
        st.write(recommendations)
    else:
        st.write("Recommended Movies:")
        for idx, title in enumerate(recommendations):
            st.write(f"{idx + 1}. {title}")


      

