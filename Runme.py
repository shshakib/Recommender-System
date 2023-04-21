from Movie import Movie
from Methods import Methods
import surprise
import os

import random
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
ratings_path = os.path.join(script_dir, 'Data', 'ratings.csv')
movies_path = os.path.join(script_dir, 'Data', 'movies.csv')

np.random.seed(85)
random.seed(85)

movie_obj = Movie(ratings_path, movies_path)
print("Loading movie dataset...")
dataset = movie_obj.load_data()

#Build dictionary for Methods
print("Building dictionary for Methods...")
methods_dict = {}
methods_dict['SVD'] = surprise.SVD()
methods_dict['Random'] = surprise.NormalPredictor()

#Run methods
print("Prepairing the dataset (train/test)...")
methods_obj = Methods(dataset, methods_dict)

print("Running algorithms on the dataset...")
methods_obj.Evaluate()
methods_obj.top_n_recommendation(movie_obj)
