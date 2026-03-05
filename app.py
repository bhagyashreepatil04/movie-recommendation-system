import streamlit as st
import pandas as pd
import ast
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 AI Movie Recommendation System")

API_KEY = "dcb1c06329d7f0016f3e2bb5c9b4e1da"

# fetch poster
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    data = requests.get(url).json()
    poster_path = data["poster_path"]
    return "https://image.tmdb.org/t/p/w500/" + poster_path

# fetch rating
def fetch_rating(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    data = requests.get(url).json()
    return data["vote_average"]

# load data
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.zip")

movies = movies.merge(credits, on="title")
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.dropna(inplace=True)

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id','title','tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

similarity = cosine_similarity(vectors)

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)),
                         reverse=True,
                         key=lambda x: x[1])[1:6]

    names = []
    posters = []
    ratings = []

    for i in movies_list:
        movie_id = new_df.iloc[i[0]].movie_id
        names.append(new_df.iloc[i[0]].title)
        posters.append(fetch_poster(movie_id))
        ratings.append(fetch_rating(movie_id))

    return names, posters, ratings


# search box
search = st.text_input("Search movie")

movie_list = new_df['title'].values

# filter movies based on search
if search:
    movie_list = [movie for movie in movie_list if search.lower() in movie.lower()]

selected_movie = st.selectbox("Select movie", movie_list)

if st.button("Recommend"):

    names, posters, ratings = recommend(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(posters[0])
        st.write(names[0])
        st.write("⭐", ratings[0])

    with col2:
        st.image(posters[1])
        st.write(names[1])
        st.write("⭐", ratings[1])

    with col3:
        st.image(posters[2])
        st.write(names[2])
        st.write("⭐", ratings[2])

    with col4:
        st.image(posters[3])
        st.write(names[3])
        st.write("⭐", ratings[3])

    with col5:
        st.image(posters[4])
        st.write(names[4])

        st.write("⭐", ratings[4])
