import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Read movies.csv and ratings.csv
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

# Assume userId is known
user_id = 1

# Step 1: Content-Based Recommendation
# Filter ratings_df for the user's ratings
user_ratings = ratings_df[ratings_df['userId'] == user_id]

# Get the top rated movie for the user
top_rated_movie_id = user_ratings.sort_values(by='rating', ascending=False).iloc[0]['movieId']

# Find the movie title for the top rated movie
top_rated_movie_title = movies_df[movies_df['movieId'] == top_rated_movie_id]['title'].iloc[0]

# Use the top rated movie title to recommend similar movies using content-based recommendation system
def content_based_recommendation(movie_title, movies_df):
    # Preprocess movie titles
    movies_df['title'] = movies_df['title'].str.lower()
    
    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit and transform TF-IDF vectors
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['title'])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get index of the movie
    idx = movies_df[movies_df['title'] == movie_title.lower()].index[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top 10 similar movies
    sim_scores = sim_scores[1:11]
    
    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return recommended movies
    return movies_df.iloc[movie_indices]['title']

content_based_recommendations = content_based_recommendation(top_rated_movie_title, movies_df)

print("Content-Based Recommendations:")
print(content_based_recommendations)

# Step 2: Collaborative Filtering Recommendation
# Create Surprise dataset
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Initialize SVD algorithm
algo = SVD()

# Train the algorithm
algo.fit(trainset)

# Predict ratings for the user
user_movies = user_ratings['movieId'].tolist()
movies_not_rated_by_user = [movie_id for movie_id in ratings_df['movieId'].unique() if movie_id not in user_movies]
user_predictions = [algo.predict(user_id, movie_id) for movie_id in movies_not_rated_by_user]

# Sort predictions by estimated rating
user_predictions.sort(key=lambda x: x.est, reverse=True)

# Get top 10 recommendations
top_collaborative_filtering_recommendations = [pred.iid for pred in user_predictions[:10]]

print("\nCollaborative Filtering Recommendations:")
print(movies_df[movies_df['movieId'].isin(top_collaborative_filtering_recommendations)]['title'])
