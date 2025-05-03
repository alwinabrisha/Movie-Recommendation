import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore

# Load the dataset (update with the correct path)
df = pd.read_csv(r"C:\Users\hp\Desktop\movie_recommended\movies.csv")

# Check for missing values
df.fillna("", inplace=True)  # Replace NaN with empty strings
df["rating"] = df["rating"].fillna("Unknown")  # Fix for inplace warning

# Select only necessary columns
df = df[['title', 'listed_in', 'description', 'cast', 'director']]

# Convert all titles to lowercase (for case-insensitive matching)
df["title"] = df["title"].str.lower().str.strip()

# Create a combined feature column
df["combined_features"] = df["listed_in"] + " " + df["description"] + " " + df["cast"] + " " + df["director"]

# Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words="english")
features_matrix = vectorizer.fit_transform(df["combined_features"])

# Compute similarity scores
similarity_matrix = cosine_similarity(features_matrix)

# ✅ Movie Recommendation Function
def recommend_movie(movie_title):
    movie_title = movie_title.lower().strip()  # Convert input to lowercase

    if movie_title not in df["title"].values:
        return "Movie not found! Try another title."

    # Get movie index
    idx = df[df["title"] == movie_title].index[0]

    # Get similarity scores
    scores = list(enumerate(similarity_matrix[idx]))

    # Sort movies based on similarity score
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    # Get recommended movie titles
    recommended_movies = [df.iloc[i[0]]["title"] for i in sorted_scores]

    return recommended_movies

# ✅ Example usage
movie_name = "Breaking Bad"  # Change this to test other movies

if movie_name.lower() in df["title"].values:
    print(f'"{movie_name}" is present in the dataset.')
    print(f"Recommended movies: {recommend_movie(movie_name)}")
else:
    print(f'"{movie_name}" is NOT found in the dataset. Try another movie title.')
import pandas as pd

# Load Movies Data
movies_df = pd.read_csv(r"C:\Users\hp\Desktop\movie_recommended\movies.csv")

# Load Ratings Data
ratings_df = pd.read_csv(r"C:\Users\hp\Desktop\movie_recommended\ratings.csv")

# Display the first few rows
print("Movies Data:")
print(movies_df.head())
print("\nRatings Data:")
print(ratings_df.head())
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load Ratings Data
ratings = pd.read_csv("ratings.csv")

# Define Reader Object (specify rating scale)
reader = Reader(rating_scale=(0.5, 5.0))

# Load Data into Surprise Dataset
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split Data into Train and Test
trainset, testset = train_test_split(data, test_size=0.2)

# Use SVD Algorithm (Singular Value Decomposition)
model = SVD()

# Train the model
model.fit(trainset)

# Predict on the test set
predictions = model.test(testset)

# Evaluate RMSE (Root Mean Squared Error)
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse}")
# Example usage: Test the recommendation system
movie_name = "Breaking Bad"  # You can change this to any other movie to test
if movie_name.lower() in df["title"].values:
    print(f'"{movie_name}" is present in the dataset.')
    print(f"Recommended movies: {recommend_movie(movie_name)}")
else:
    print(f'"{movie_name}" is NOT found in the dataset. Try another movie title.')
# List of movie names to test
movie_names = ["The Godfather", "The Dark Knight", "Inception", "Stranger Things", "Game of Thrones"]

# Loop through each movie name and test recommendations
for movie_name in movie_names:
    if movie_name.lower() in df["title"].values:
        print(f'"{movie_name}" is present in the dataset.')
        print(f"Recommended movies: {recommend_movie(movie_name)}")
    else:
        print(f'"{movie_name}" is NOT found in the dataset. Try another movie title.')
import os

# Check if the files exist before loading them
movies_file_path = r"C:\Users\hp\Desktop\movie_recommended\movies.csv"
ratings_file_path = r"C:\Users\hp\Desktop\movie_recommended\ratings.csv"

# Check if files exist
if os.path.exists(movies_file_path) and os.path.exists(ratings_file_path):
    print("Both files found. Proceeding to load the data...")
    
    # Load Movies Data
    df = pd.read_csv(movies_file_path)
    
    # Load Ratings Data
    ratings_df = pd.read_csv(ratings_file_path)
    
    # Display the first few rows
    print("Movies Data:")
    print(df.head())
    print("\nRatings Data:")
    print(ratings_df.head())
else:
    print("Error: One or both files are missing. Please check the file paths.")
