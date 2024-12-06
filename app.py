from flask import Flask, render_template, request, jsonify
import pandas as pd
from cassandra.cluster import Cluster
# Start spark session
from pyspark.sql import SparkSession, Row
from pyspark.ml.recommendation import ALSModel
import random

spark = SparkSession.builder.config("spark.driver.memory", "10g").getOrCreate()
# Load the model
als_model = ALSModel.load("models/als_model")

app = Flask(__name__)

# Configure Cassandra connection
cluster = Cluster(['172.19.0.2'])  # Replace with your Cassandra nodes
session = cluster.connect('anime')  # Replace 'your_keyspace' with your keyspace

# Load dataset
df_anime = pd.read_csv('dataset/myanimelist-dataset/processed-dataset/anime-dataset-2023.csv')
df_anime = df_anime[~df_anime['Genres'].str.contains('UNKNOWN')]
df_anime = df_anime[~df_anime['Genres'].str.contains('Hentai')]
df_anime = df_anime[~df_anime['Studios'].str.contains('UNKNOWN')]

df_user = pd.read_csv('dataset/myanimelist-dataset/processed-dataset/users-details-2023.csv')

anime_id_list = df_anime['anime_id'].values
name_list = df_anime['Name'].values
english_name_list = df_anime['English name'].values

anime_reduced_data = [{'anime_id': id, 'name': name, 'english_name': english_name} for id, name, english_name in zip(anime_id_list, name_list, english_name_list)]
del anime_id_list, name_list, english_name_list

# Home route
@app.route('/')
def home():
    return render_template('index.html')


def collaborative_filtering(user_id: int, number_of_recommendations: int):
    # convert user id to Dataframe
    user = spark.createDataFrame([Row(user_id=user_id)])
    # get recommendations
    recommendations = als_model.recommendForUserSubset(user, 100)
    # convert recommendations from Pyspark Dataframe to Pandas Dataframe
    recommendation_pd = recommendations.toPandas()
    if recommendation_pd.empty:
        return None
    
    # Get recommended anime ids
    recommend_ids = [recommendation['anime_id'] for recommendation in recommendation_pd['recommendations'][0]]

    recommended_df = df_anime[df_anime['anime_id'].isin(recommend_ids)][:number_of_recommendations]
    return recommended_df


def content_based_filtering(anime_name: str, number_of_recommendations: int):
    anime_name = anime_name.lower()
    filtered_animes = [anime for anime in anime_reduced_data if anime_name in anime['name'].lower() or (anime['english_name'] != 'UNKNOWN' and anime_name in anime['english_name'].lower())]
    if not filtered_animes:
        return None
    
    chosen_anime = filtered_animes[0]
    query = f"SELECT * FROM anime_similarity where anime_id={chosen_anime['anime_id']}"  # Replace with your query
    rows = session.execute(query)
    result = []
    for row in rows:
        result.append({
            'anime_id': row.anime_id_2,  # Replace with your columns
            'cos_sim': row.cos_sim
        })
    
    result.sort(reverse=True, key=lambda recommended_anime: recommended_anime['cos_sim'])
    result = result[:number_of_recommendations]
    recommend_ids = [recommended_anime['anime_id'] for recommended_anime in result]
    recommended_df = df_anime[df_anime['anime_id'].isin(recommend_ids)]
    return recommended_df


def random_filtering(number_of_recommendations: int):
    anime_id_list = [anime['anime_id'] for anime in anime_reduced_data]
    random_ids = random.sample(anime_id_list, number_of_recommendations)
    recommended_df = df_anime[df_anime['anime_id'].isin(random_ids)]
    return recommended_df


# Recommendation route
@app.route('/recommend', methods=['POST'])
def recommend():
    recommendation_type = request.form['recommendation_type']
    num_recommendations = int(request.form['num_recommendations'])

    if recommendation_type == 'content_based':
        anime_name = request.form['anime_name']
        if not anime_name:
            return render_template('index.html', error_message="Please enter Anime name.", recommendation_type=recommendation_type)
        
        recommended_animes = content_based_filtering(anime_name, num_recommendations)
        if recommended_animes is None or recommended_animes.empty:
            message2 = "Anime " + str(anime_name) + " does not exist"
            return render_template('recommendations.html', message=message2, animes=None, recommendation_type=recommendation_type)
        
        return render_template('recommendations.html', animes=recommended_animes, recommendation_type=recommendation_type)
    elif recommendation_type == 'collaborative':
        user_id = request.form['user_id']
        
        if not user_id:
            return render_template('index.html', error_message="Please enter a User ID.", recommendation_type=recommendation_type)
        try:
            user_id = int(user_id)
        except ValueError:
            return render_template('index.html', error_message="Please enter a valid User ID (must be an integer).", recommendation_type=recommendation_type)
        
        recommended_animes = collaborative_filtering(user_id, num_recommendations)
        if recommended_animes is None or recommended_animes.empty:
            recommended_animes = random_filtering(num_recommendations)
            return render_template('recommendations.html', animes=recommended_animes, recommendation_type=recommendation_type)

            # message2 = "No recommendations found"
            # return render_template('recommendations.html', message=message2, animes=None, recommendation_type=recommendation_type)
        
        return render_template('recommendations.html', animes=recommended_animes, recommendation_type=recommendation_type)

    return render_template('index.html', error_message="Please select a recommendation type.")


# New route to handle anime name autocomplete
@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    search_term = request.args.get('term')
    search_term = search_term.lower()
    anime_names = []
    if search_term:
        filtered_animes = [anime for anime in anime_reduced_data if search_term in anime['name'].lower() or (anime['english_name'] != 'UNKNOWN' and search_term in anime['english_name'].lower())]
        anime_names = [filtered_anime['name'] for filtered_anime in filtered_animes]
        anime_names.extend([filtered_anime['english_name'] for filtered_anime in filtered_animes])
    # else:
    #     anime_names = df_anime['Name'].tolist()
    # print(anime_names)
    return jsonify(anime_names)

if __name__ == '__main__':
    app.run(debug=True)