# Recommender-System

•	Movie: This class reads data from CSV files and returns movie names and genres based on movie ID.
•	DataSlicer: This class splits the data into train and test sets using a random state.
•	Methods: This class evaluated a dictionary list of methods. It is also returned the top_n_recommendation movies for a given user. 
•	Metrics: This class calculated the evaluation metrics for each method, such as MAE, RMSE, and hit rate.
•	ContentBased: This class inherited from the AlgoBase class of the surprise package and implemented a content-based filtering method using cosine similarity between movie genres.
•	Hybrid: This class also inherited from the AlgoBase class of the surprise package and implemented a hybrid filtering method using a weighted average of the ratings from collaborative filtering, SVD and content-based filtering.
•	RunMe: This file created objects of the above classes and ran the main program. We can pass compute argument to load predictions from previously saved results without computing similarity matrix again.






## References
Bart Baesens, S. v. (2020). Item-Item Collaborative Filtering: a Refresher. Retrieved from datamining apps: https://www.dataminingapps.com/2020/01/item-item-collaborative-filtering-a-refresher/
Cosine similarity. (n.d.). Retrieved from Wikipedia: https://en.wikipedia.org/wiki/Cosine_similarity#:~:text=Cosine%20similarity%20is%20the%20cosine,the%20product%20of%20their%20lengths.
Harper, F. M. (2015). The MovieLens Datasets: History and Context. ACM Trans. Interact. Intell. Syst., 19.
Hug, N. (n.d.). Matrix Factorization-based algorithms. Retrieved from Surprise’ documentation!: https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD
Kane, F. (2020). Building Recommender Systems with Machine Learning and AI. Retrieved from Linkedin Learning: https://www.linkedin.com/learning/building-recommender-systems-with-machine-learning-and-ai/install-anaconda-review-course-materials-and-create-movie-recommendations
Mr.Avadhut D.Wagavkar, P. M. (2017). Weighted Hybrid Approach in Recommendation Method. International Journal of Computer Science Trends and Technology (IJCST), 5.
University, S. (2016). Collaborative Filtering. Retrieved from Youtube: https://www.youtube.com/watch?v=h9gpufJFF-0
University, S. (2016). Latent Factor Recommender System. Retrieved from Youtube: https://www.youtube.com/watch?v=E8aMcwmqsTg

