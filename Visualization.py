import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset (Ensure correct absolute file path)
try:
    df = pd.read_csv(r'C:\Users\Administrator\Desktop\movie_recommendation\movies.csv')  # Absolute path
    print(df.head())  # Print the first few rows to check if it's loaded correctly
    print("\nColumn Names:", df.columns)  # Print the column names to inspect
except FileNotFoundError:
    print("The file 'movies.csv' was not found. Please check the file path.")
    exit()  # Exit the program if the file isn't found

# Assuming the column names based on inspection
# You can adjust the following based on the column names in your dataset
# Example: If the actual column names are 'type' and 'description', adjust accordingly.

# Check the columns to ensure the relevant data exists
if 'type' in df.columns and 'description' in df.columns:
    # Combine relevant text fields into one for content-based filtering
    df['combined'] = df['type'].fillna('') + " " + df['description'].fillna('')

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined'])

    # Cosine Similarity Matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Normalize titles (Assuming the 'title' column exists)
    if 'title' in df.columns:
        df['normalized_title'] = df['title'].str.lower().str.strip()

    # Recommendation Function
    def recommend(title, num_recommendations=5):
        title = title.lower().strip()

        # Match movie titles
        matches = df[df['normalized_title'].str.contains(title)]
        if matches.empty:
            return f"❌ Movie containing '{title}' not found in dataset."

        idx = matches.index[0]  # Pick the first match
        similarity_scores = list(enumerate(cosine_sim[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
        movie_indices = [i[0] for i in similarity_scores]

        recommended = df.iloc[movie_indices][['title', 'type']]  # Adjust column names accordingly
        return recommended

    # Example usage
    movie_to_search = "Batman"  # Try 'hero', 'war', 'spider' etc. if 'Batman' is not in dataset
    print(f"\n🎬 Recommendations for '{movie_to_search}':\n")
    result = recommend(movie_to_search)

    # Output the result
    if isinstance(result, str):
        print(result)
    else:
        for i, row in result.iterrows():
            print(f"➡️ {row['title']} [{row['type']}]")  # Adjust column names accordingly
else:
    print("Required columns (e.g., 'type', 'description') not found in the dataset.")


