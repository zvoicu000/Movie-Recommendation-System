from flask import Flask, render_template, request
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load movies data
movies_data = pd.read_csv('data/movies.csv')

# Select relevant features for recommendation
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# Replace null values with empty strings
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine selected features into a single string
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Convert text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Calculate cosine similarity
similarity = cosine_similarity(feature_vectors)

# Home and Recommendation route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        movie_name = request.form['movie_name']
        list_of_all_titles = movies_data['title'].tolist()
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
        if not find_close_match:
            return render_template('index.html', movie_name=movie_name, movies=[], not_found=True)
        close_match = find_close_match[0]
        index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        recommended_movies = [movies_data[movies_data.index == movie[0]]['title'].values[0] for movie in sorted_similar_movies[1:31]]
        return render_template('index.html', movie_name=movie_name, movies=recommended_movies, not_found=False)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
