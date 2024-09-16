from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import svds

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random string for security

# Load dataset
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Create the user-item matrix
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
matrix = user_movie_matrix.to_numpy()
user_ratings_mean = np.mean(matrix, axis=1)
matrix_normalized = matrix - user_ratings_mean.reshape(-1, 1)
U, sigma, Vt = svds(matrix_normalized, k=50)
sigma = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_movie_matrix.columns)

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('profile'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form['user_id']
        if user_id and user_id.isdigit() and int(user_id) in user_movie_matrix.index:
            session['user_id'] = int(user_id)
            return redirect(url_for('profile'))
        else:
            return "Invalid User ID", 400
    return render_template('login.html')

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('profile.html', user_id=session['user_id'])

@app.route('/index')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = int(session['user_id'])
    num_recommendations = int(request.form['num_recommendations'])
    
    def recommend_movies(user_id, num_recommendations=5):
        try:
            user_row = predicted_ratings_df.iloc[user_id - 1].sort_values(ascending=False)
            user_rated_movies = ratings[ratings.userId == user_id]['movieId']
            recommendations = user_row.drop(user_rated_movies).head(num_recommendations)
            return movies[movies.movieId.isin(recommendations.index)]
        except IndexError:
            return pd.DataFrame()

    recommended_movies = recommend_movies(user_id=user_id, num_recommendations=num_recommendations)
    return render_template('recommendations.html', movies=recommended_movies.to_dict(orient='records'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
