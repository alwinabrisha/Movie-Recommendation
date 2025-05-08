import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load your datasets (replace with your file paths)
movies_df = pd.read_csv('movies.csv')  # Replace with your movies dataset path
ratings_df = pd.read_csv('ratings.csv')  # Replace with your ratings dataset path

# Debugging: Check the column names and sample data
print(ratings_df.columns)
print(ratings_df.head())

# Step 2: Clean and rename columns if necessary
ratings_df.columns = ratings_df.columns.str.strip()  # Strip any spaces
ratings_df.rename(columns={'SuserId': 'userId', 'movieId': 'movieId', 'rating': 'rating', 'timestamp': 'timestamp'}, inplace=True)

# Step 3: Create a User-Item Matrix (Pivot Table)
user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Step 4: Convert user-item matrix into an array for use in KNN
X = user_item_matrix.values

# Step 5: Train a KNN model for recommendations
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(X)

# Step 6: Make predictions (For example, predict for user 1)
user_index = 0  # Example: predicting for the first user (userId = 1)
distances, indices = model.kneighbors(X[user_index].reshape(1, -1), n_neighbors=5)

# Display recommended movie IDs for the user
print(f"Recommended movie IDs for user {user_index + 1}: {indices[0]}")

# Step 7: Get actual and predicted ratings for the first user
predictions = X[user_index, indices[0]]
actual_ratings = X[user_index, indices[0]]

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(actual_ratings, predictions))
print(f"RMSE for KNN Model: {rmse}")

# Step 8: Visualize Predicted vs Actual Ratings
plt.scatter(actual_ratings, predictions, color='blue', label='Predicted vs Actual')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Predicted vs Actual Ratings for User 1')
plt.legend()
plt.show()

# Step 9: If you have specific movies to see the actual movie names
recommended_movie_titles = movies_df.iloc[indices[0]]['title']
print("Recommended Movies for User 1:")
print(recommended_movie_titles)


