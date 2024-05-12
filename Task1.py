import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Read movies.csv
movies_df = pd.read_csv('movies.csv')

# Read ratings.csv
ratings_df = pd.read_csv('ratings.csv')

# Content-Based Recommendation System
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

# Collaborative Filtering Recommendation System
def collaborative_filtering_recommendation(user_id, ratings_df):
    # Create Surprise dataset
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    
    # Split data into train and test sets
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # Initialize SVD algorithm
    algo = SVD()
    
    # Train the algorithm
    algo.fit(trainset)
    
    # Predict ratings for the test set
    predictions = algo.test(testset)
    
    # Evaluate accuracy (optional)
    accuracy.rmse(predictions)
    
    # Get top N movie recommendations for the user
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid == user_id:
            if iid not in top_n:
                top_n[iid] = est
    top_movie_ids = sorted(top_n, key=top_n.get, reverse=True)[:10]
    
    # Return recommended movies
    return movies_df[movies_df['movieId'].isin(top_movie_ids)]['title']

# Test content-based recommendation
movie_title = 'Toy Story'
print("Content-Based Recommendation for '{}'".format(movie_title))
print(content_based_recommendation(movie_title, movies_df))

# Test collaborative filtering recommendation
user_id = 1
print("\nCollaborative Filtering Recommendation for User {}".format(user_id))
print(collaborative_filtering_recommendation(user_id, ratings_df))
