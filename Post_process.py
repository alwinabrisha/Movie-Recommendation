import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the movies metadata CSV (Netflix-style)
movies_df = pd.read_csv("movies.csv")

# Sample user-movie ratings matrix (you may already have this)
# For demo, we simulate users and the titles they watched
user_watch_history = {
    1: ["Kota Factory", "Midnight Mass"],
    2: ["Ganglands", "Midnight Mass"],
    3: ["Kota Factory", "Blood & Water"],
    4: ["Dick Johnson Is Dead", "Sankofa"],
    5: ["Jailbirds New Orleans", "Blood & Water"],
    266: ["My Little Pony: A New Generation", "Kota Factory"],
    310: ["Midnight Mass", "Sankofa"],
    408: ["Ganglands", "Kota Factory"],
    510: ["Blood & Water", "Midnight Mass"]
}

# Convert the watch history into a user-item matrix
all_titles = list(movies_df['title'].dropna().unique())
user_ids = list(user_watch_history.keys())

# Build binary matrix: 1 if user watched the movie, else 0
data = []
for uid in user_ids:
    row = [1 if title in user_watch_history[uid] else 0 for title in all_titles]
    data.append(row)

ratings_matrix = pd.DataFrame(data, index=user_ids, columns=all_titles)

# Calculate cosine similarity between users
similarity_matrix = cosine_similarity(ratings_matrix)

# Convert to DataFrame for readability
similarity_df = pd.DataFrame(similarity_matrix, index=user_ids, columns=user_ids)

# Choose a target user to recommend for
target_user = 1

# Get top 5 similar users to target_user (excluding themselves)
similar_users = similarity_df[target_user].sort_values(ascending=False)[1:6].index.tolist()

print(f"\nTop 5 similar users to userId={target_user} and their watched movies:\n")

for i, uid in enumerate(similar_users, start=1):
    watched = user_watch_history[uid]
    watched_titles = movies_df[movies_df['title'].isin(watched)]['title']
    print(f"Neighbor {i}: UserId {uid}")
    print("Watched Movies:")
    for title in watched_titles:
        print(f"- {title}")
    print()

