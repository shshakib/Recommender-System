from DataSlicer import DataSlicer
from Metrics import Metrics

class Methods:

    def __init__(self, dataset, methods_dict):
        self.sliced_dataset = DataSlicer(dataset)
        self.methods_dict = methods_dict

        
    def Evaluate(self, n=10):
        #Calculate RMSE/MAE
        metrics = {}
        result = {}
        for method_name, method_obj in self.methods_dict.items():
            print("Evaluating ", method_name, "...")
            method_obj.fit(self.sliced_dataset.get_train_dataset())
            predictions = method_obj.test(self.sliced_dataset.get_test_dataset())
            metrics["RMSE"] = Metrics.RMSE(predictions)
            metrics["MAE"] = Metrics.MAE(predictions)

            #Leave one out
            method_obj.fit(self.sliced_dataset.get_leave_one_out_train_dataset())
            leave_one_predictions = method_obj.test(self.sliced_dataset.get_leave_one_out_test_dataset())        
            all_predictions = method_obj.test(self.sliced_dataset.get_leave_one_out_antitest_dataset())
            #Top n recommentaion for each user
            top_n_predicted = Metrics.get_top_n(all_predictions, n)
            metrics["HR"] = Metrics.hit_rate(top_n_predicted, leave_one_predictions)  

            result[method_name] = metrics.copy()
        print("{:<10} {:<10} {:<10} {:<10}".format("Methods", "RMSE", "MAE", "HR"))
        for (method, value) in result.items():
            print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f}".format(method, value["RMSE"], value["MAE"], value["HR"]))

        return metrics
    
    def top_n_recommendation(self, movie_obj, test_user=85, k=10):
        
        for method_name, method_obj in self.methods_dict.items():
            print("\nBuilding recommendation model for ", method_name, " ...")
            train_dataset = self.sliced_dataset.get_full_train_dataset()
            method_obj.fit(train_dataset)
            
            test_dataset = self.sliced_dataset.get_user_anti_test_dataset(test_user)
            predictions = method_obj.test(test_dataset)
            
            recommendations = []
            print ("\nRecommendation list:")
            for user_id, movie_id, r_ui, estimated_rating, _ in predictions:
                intMovieID = int(movie_id)
                recommendations.append((intMovieID, estimated_rating))
            
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            for ratings in recommendations[:10]:
                print(movie_obj.get_movie_name(ratings[0]), ratings[1])
    
    