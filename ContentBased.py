import surprise
from Movie import Movie
from tqdm import tqdm

import math
import numpy as np
import heapq

class ContentBased(surprise.AlgoBase):

    def __init__(self, movie_obj, knn=40, sim_options={}):
        surprise.AlgoBase.__init__(self)
        self.knn = knn
        self.genres = movie_obj.get_movie_genres()


    def fit(self, train_dataset):
        surprise.AlgoBase.fit(self, train_dataset)
        #print("Bulding content-based similarity matrix...")   
        self.similarities = np.zeros((train_dataset.n_items, train_dataset.n_items))
        for movie_i in tqdm(range(train_dataset.n_items)):
            for movie_j in range(movie_i+1, train_dataset.n_items):
                
                #Calculate similarity
                #genre_similarity = self.genre_similarity(int(train_dataset.to_raw_iid(movie_i)), int(train_dataset.to_raw_iid(movie_j)), self.genres)
                # genres_i = self.genres[int(train_dataset.to_raw_iid(movie_i))]
                # genres_j = self.genres[int(train_dataset.to_raw_iid(movie_j))]
                # sum_ii = np.dot(genres_i, genres_i)
                # sum_jj = np.dot(genres_j, genres_j)
                # sum_ij = np.dot(genres_i, genres_j)
                # genre_similarity = sum_ij/math.sqrt(sum_ii * sum_jj)
                # self.similarities[movie_i, movie_j] = genre_similarity
                # self.similarities[movie_j, movie_i] = genre_similarity
                genres_i = self.genres[int(train_dataset.to_raw_iid(movie_i))]
                genres_j = self.genres[int(train_dataset.to_raw_iid(movie_j))]
                sum_ii, sum_ij, sum_jj = 0, 0, 0
                for idx in range(len(genres_i)):
                    i = genres_i[idx]
                    j = genres_j[idx]
                    sum_ii += i * i
                    sum_jj += j * j
                    sum_ij += i * j

        return self
    

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise surprise.PredictionImpossible('User and/or item is unkown.')
        
        #Similarity between this item and everything the user rated
        neighbors = []
        for rating in self.trainset.ur[u]:
            neighbors.append( (self.similarities[i,rating[0]], rating[1]) )
        
        #Top-K most-similar ratings
        k_neighbors = heapq.nlargest(self.knn, neighbors, key=lambda t: t[0])
        
        #Average similarity score of K neighbors weighted by user ratings
        sim_total = weighted_sum = 0
        for (sim_score, rating) in k_neighbors:
            if (sim_score > 0):
                sim_total += sim_score
                weighted_sum += sim_score * rating
            
        if (sim_total == 0):
            raise surprise.PredictionImpossible('No neighbors')

        predicted_rating = weighted_sum / sim_total

        return predicted_rating
    