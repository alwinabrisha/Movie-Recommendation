import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("movies.csv")

# Fill missing values
df.fillna("", inplace=True)

# 🔹 1. Genre Frequency Plot
plt.figure(figsize=(10, 6))
all_genres = df['listed_in'].str.split(', ').explode()
top_genres = all_genres.value_counts().head(10)
sns.barplot(x=top_genres.values, y=top_genres.index, palette="viridis")
plt.title("Top 10 Genres")
plt.xlabel("Number of Movies")
plt.ylabel("Genre")
plt.tight_layout()
plt.show()

# 🔹 2. Movie Type Count (TV Show vs Movie)
plt.figure(figsize=(6, 4))
sns.countplot(x="type", data=df, palette="pastel")
plt.title("Content Type Count")
plt.xlabel("Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 🔹 3. Movie Release Years Distribution (If column exists)
if 'release_year' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df['release_year'], bins=30, kde=False, color="skyblue")
    plt.title("Movies by Release Year")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# 🔹 4. Top 10 Most Frequent Directors
top_directors = df['director'].str.split(', ').explode().value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_directors.values, y=top_directors.index, palette="magma")
plt.title("Top 10 Directors")
plt.xlabel("Number of Movies")
plt.ylabel("Director")
plt.tight_layout()
plt.show()

# 🔹 5. Top 10 Actors (optional if 'cast' column is detailed)
top_actors = df['cast'].str.split(', ').explode().value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_actors.values, y=top_actors.index, palette="coolwarm")
plt.title("Top 10 Appearing Actors")
plt.xlabel("Appearances")
plt.ylabel("Actor")
plt.tight_layout()
plt.show()