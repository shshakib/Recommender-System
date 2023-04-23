import surprise
import numpy as np
import pandas as pd

class Movie:

    def __init__(self, ratings_csv_path, movies_csv_path):
        self.ratings_path = ratings_csv_path
        self.movies_path = movies_csv_path


    def load_data(self):
        reader = surprise.Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
        data = surprise.Dataset.load_from_file(self.ratings_path, reader=reader)

        return data
    

    # def get_user_ratings(self, user_id):
    #      ratings_df = pd.read_csv(self.ratings_path)
    #      user_ratings = ratings_df[ratings_df['userId'] == user_id]

    #      return list(zip(user_ratings['movieId'], user_ratings['rating']))

   
    # def get_movie_popularity(self):
    #     ratings_df = pd.read_csv(self.ratings_path)
    #     movie_popularity = ratings_df.groupby('movieId').size().sort_values(ascending=False)
    #     movie_ranking = dict(zip(movie_popularity.index, range(1, len(movie_popularity) + 1)))

    #     return movie_ranking
    

    def get_movie_genres(self):
        movies_df = pd.read_csv(self.movies_path, usecols=['movieId', 'genres'])
        self.movie_list = movies_df['movieId'].tolist()
        all_genres = sorted(set('|'.join(movies_df['genres']).split('|')))
        movielens_data = movies_df.to_dict('records')
        movie_genre_dict = {}
        for row in movielens_data:
            movie_id = row['movieId']
            genres = row['genres'].split('|')
            genre_list = [0] * len(all_genres)
            for genre in genres:
                genre_index = all_genres.index(genre)
                genre_list[genre_index] = 1
            movie_genre_dict[movie_id] = genre_list
        return movie_genre_dict
    

    def get_movie_name(self, movie_Id):
        movies_df = pd.read_csv(self.movies_path)
        movieId_to_name = pd.Series(movies_df.title.values, index=movies_df.movieId).to_dict()
        if movie_Id in movieId_to_name:
            return movieId_to_name[movie_Id]
        else:
            return ""
        

    # def get_movieId(self, movie_name):
    #     movies_df = pd.read_csv(self.movies_path)
    #     name_to_movieId = pd.Series(movies_df.movieId.values, index=movies_df.title).to_dict()       
    #     if movie_name in name_to_movieId:
    #         return name_to_movieId[movie_name]
    #     else:
    #         return 0