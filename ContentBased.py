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
                genre_similarity = self.genre_similarity(int(train_dataset.to_raw_iid(movie_i)), int(train_dataset.to_raw_iid(movie_j)), self.genres)
                self.similarities[movie_i, movie_j] = genre_similarity
                self.similarities[movie_j, movie_i] = genre_similarity

        return self
    
    def genre_similarity(self, movie_i, movie_j, genres):
        genres_i = genres[movie_i]
        genres_j = genres[movie_j]
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(genres_i)):
            x = genres_i[i]
            y = genres_j[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        
        return sumxy/math.sqrt(sumxx*sumyy)
    import numpy as np

    # def genre_similarity(self, movie_i, movie_j, genres):
    #     genres_i = genres[movie_i]
    #     genres_j = genres[movie_j]
    #     sumxx = np.dot(genres_i, genres_i)
    #     sumyy = np.dot(genres_j, genres_j)
    #     sumxy = np.dot(genres_i, genres_j)
    
    #     return sumxy / np.sqrt(sumxx * sumyy)
    

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise surprise.PredictionImpossible('User and/or item is unkown.')
        
        # Build up similarity scores between this item and everything the user rated
        neighbors = []
        for rating in self.trainset.ur[u]:
            genreSimilarity = self.similarities[i,rating[0]]
            neighbors.append( (genreSimilarity, rating[1]) )
        
        # Extract the top-K most-similar ratings
        k_neighbors = heapq.nlargest(self.knn, neighbors, key=lambda t: t[0])
        
        # Compute average sim score of K neighbors weighted by user ratings
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbors:
            if (simScore > 0):
                simTotal += simScore
                weightedSum += simScore * rating
            
        if (simTotal == 0):
            raise surprise.PredictionImpossible('No neighbors')

        predictedRating = weightedSum / simTotal

        return predictedRating
    