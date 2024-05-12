import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate, train_test_split
from surprise.accuracy import rmse, mae
from surprise.prediction_algorithms.knns import KNNBasic

# Read movies.csv and ratings.csv
movies_df = pd.read_csv('movies.csv')
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
    return movies_df.iloc[movie_indices]['movieId'].tolist()

# Collaborative Filtering Recommendation System
def collaborative_filtering_recommendation(user_id, ratings_df):
    # Create Surprise dataset
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    
    # Train-test split
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # Initialize SVD algorithm
    algo = SVD()
    
    # Cross-validate the algorithm
    cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    
    # Train the algorithm on the full dataset
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    
    # Predict ratings for the test set
    predictions = algo.test(testset)
    
    # Compute RMSE and MAE
    rmse_score = rmse(predictions)
    mae_score = mae(predictions)
    
    # Get top N movie recommendations for the user
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid == user_id:
            if iid not in top_n:
                top_n[iid] = est
    top_movie_ids = sorted(top_n, key=top_n.get, reverse=True)[:10]
    
    # Return recommended movies
    return top_movie_ids, cv_results['test_rmse'].mean(), cv_results['test_mae'].mean()

# Assume user_id is known
user_id = 1
# Assume a movie title is known
movie_title = 'Toy Story'

# Evaluate Content-Based Recommendation System
content_based_recommendations = content_based_recommendation(movie_title, movies_df)

# Evaluate Collaborative Filtering Recommendation System
cf_recommendations, avg_rmse, avg_mae = collaborative_filtering_recommendation(user_id, ratings_df)

# Print results
print("Content-Based Recommendations:", content_based_recommendations)
print("Average RMSE (Collaborative Filtering):", avg_rmse)
print("Average MAE (Collaborative Filtering):", avg_mae)
print("Collaborative Filtering Recommendations:", cf_recommendations)
