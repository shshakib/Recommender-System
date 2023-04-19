import surprise

class DataSlicer:
    
    def __init__(self, dataset):
        
        #train/test split(80/20)
        self.train_dataset, self.test_dataset = surprise.model_selection.train_test_split(dataset, test_size=.20, random_state=85)

        #Full training set
        self.full_train_dataset = dataset.build_full_trainset()
        
        #The ratings are all the ratings that are not in the trainset, i.e. all the ratings 
        # where the user is known, the item  is known, but the rating is not in the trainset.
        self.full_antitest_dataset = self.full_train_dataset.build_anti_testset()
     
        #Leave one out train/test
        leave_one_out = surprise.model_selection.LeaveOneOut(n_splits=1, random_state=85)
        for train, test in leave_one_out.split(dataset):
            self.leave_one_out_train = train
            self.leave_one_out_test = test
        
        #anti-test-set
        self.leave_one_out_antitest_dataset = self.leave_one_out_train.build_anti_testset()
            
    def get_full_train_dataset(self):
        return self.full_train_dataset
    
    def get_full_antitest_dataset(self):
        return self.full_antitest_dataset
 
    def get_train_dataset(self):
        return self.train_dataset
    
    def get_test_dataset(self):
        return self.test_dataset
    
    def get_leave_one_out_train_dataset(self):
        return self.leave_one_out_train
    
    def get_leave_one_out_test_dataset(self):
        return self.leave_one_out_test
    
    def get_leave_one_out_antitest_dataset(self):
        return self.leave_one_out_antitest_dataset

    def get_user_anti_test_dataset(self, test_user):
        #list of all movie IDs
        movies = self.full_train_dataset().all_items()
        #list of movies that the test user has already rated
        rated_movies = [item for (item, rating) in self.full_train_dataset.ur[test_user]]
        #list of movies that the test user has not yet rated
        anti_testset = [(test_user, item, 0) for item in movies if item not in rated_movies]

        return anti_testset

