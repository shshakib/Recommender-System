from Movie import Movie
from Methods import Methods
from ContentBased import ContentBased
from Hybrid import Hybrid
import surprise
import os

import random
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
ratings_path = os.path.join(script_dir, 'Data', 'ratings.csv')
movies_path = os.path.join(script_dir, 'Data', 'movies.csv')

my_seed = 85
np.random.seed(my_seed)
random.seed(my_seed)

movie_obj = Movie(ratings_path, movies_path)
print("Loading movie dataset...")
dataset = movie_obj.load_data()


#Build dictionary for Methods
print("Building dictionary for Methods...")
methods_dict = {}
methods_dict['Content-Based'] = ContentBased(movie_obj)
methods_dict['Collaborative_KNN'] =  surprise.KNNBasic(sim_options = {'name': 'cosine', 'user_based': False})
methods_dict['SVD'] = surprise.SVD()
methods_dict['Hybrid'] = Hybrid([ContentBased(movie_obj), surprise.KNNBasic(sim_options = {'name': 'cosine', 'user_based': False}), surprise.SVD()], [0.33, 0.33, 0.34])
methods_dict['Random'] = surprise.NormalPredictor()

#Run methods
print("Prepairing the dataset (train/test)...")
methods_obj = Methods(dataset, methods_dict, script_dir)

print("Running algorithms on the dataset...")
methods_obj.Evaluate(compute = True) #Set compute to False to load the precomputed predictions.
methods_obj.top_n_recommendation(movie_obj, compute = True) #Set compute to False to load the precomputed predictions.
