from DataSlicer import DataSlicer
from Metrics import Metrics
import numpy as np
import pickle
import os
import gzip

class Methods:

    def __init__(self, dataset, methods_dict, script_dir):
        self.sliced_dataset = DataSlicer(dataset)
        self.methods_dict = methods_dict
        self.cwd = script_dir

        
    def Evaluate(self, n=10, compute = True):
        #Calculate RMSE/MAE
        metrics = {}
        result = {}
        for method_name, method_obj in self.methods_dict.items():
            print("Evaluating ", method_name, "...")
            
            #Save and Loading path
            precomputed_folder = os.path.join(self.cwd, 'Precomputed')
            if not os.path.exists(precomputed_folder):
                os.makedirs(precomputed_folder)
            
            if compute == True:
                method_obj.fit(self.sliced_dataset.get_train_dataset())
                predictions = method_obj.test(self.sliced_dataset.get_test_dataset())
                file_path = os.path.join(precomputed_folder, method_name + '_predictions.pkl.gz')
                with gzip.open(file_path, 'wb') as f:
                    pickle.dump(predictions, f)
                metrics["RMSE"] = Metrics.RMSE(predictions)
                metrics["MAE"] = Metrics.MAE(predictions)

                #Leave one out
                method_obj.fit(self.sliced_dataset.get_leave_one_out_train_dataset())
                leave_one_predictions = method_obj.test(self.sliced_dataset.get_leave_one_out_test_dataset())        
                all_predictions = method_obj.test(self.sliced_dataset.get_leave_one_out_antitest_dataset())
                file_path = os.path.join(precomputed_folder, method_name + '_leave_one_predictions.pkl.gz')
                with gzip.open(file_path, 'wb') as f:
                    pickle.dump(leave_one_predictions, f)
                file_path = os.path.join(precomputed_folder, method_name + '_all_predictions.pkl.gz')
                with gzip.open(file_path, 'wb') as f:
                    pickle.dump(all_predictions, f)
                
                #Top n recommentaion for each user
                top_n_predicted = Metrics.get_top_n(all_predictions, n)
                metrics["HR"] = Metrics.hit_rate(top_n_predicted, leave_one_predictions)
            else:
                file_path = os.path.join(precomputed_folder, method_name + '_predictions.pkl.gz')
                with gzip.open(file_path, 'rb') as f:
                    predictions = pickle.load(f)
                metrics["RMSE"] = Metrics.RMSE(predictions)
                metrics["MAE"] = Metrics.MAE(predictions)

                file_path = os.path.join(precomputed_folder, method_name + '_all_predictions.pkl.gz')
                with gzip.open(file_path, 'rb') as f:
                    all_predictions = pickle.load(f)
                
                file_path = os.path.join(precomputed_folder, method_name + '_leave_one_predictions.pkl.gz')
                with gzip.open(file_path, 'rb') as f:
                    leave_one_predictions = pickle.load(f)
                top_n_predicted = Metrics.get_top_n(all_predictions, n)
                metrics["HR"] = Metrics.hit_rate(top_n_predicted, leave_one_predictions)

            result[method_name] = metrics.copy()
        print("{:<40} {:<10} {:<10} {:<10}".format("Methods", "RMSE", "MAE", "HR"))
        for (method, value) in result.items():
            print("{:<40} {:<10.4f} {:<10.4f} {:<10.4f}".format(method, value["RMSE"], value["MAE"], value["HR"]))

        return metrics

    
    def top_n_recommendation(self, movie_obj, test_user=85, compute = True, n=10):
        
        for method_name, method_obj in self.methods_dict.items():
            
            #Save and Loading path
            precomputed_folder = os.path.join(self.cwd, 'Precomputed')
            if not os.path.exists(precomputed_folder):
                os.makedirs(precomputed_folder)

            if compute == True:
                print("\nBuilding recommendation model for", method_name, " ...")
                train_dataset = self.sliced_dataset.get_full_train_dataset()
                method_obj.fit(train_dataset)
                test_dataset = self.sliced_dataset.get_user_anti_test_dataset(test_user)
                predictions = method_obj.test(test_dataset)
                file_path = os.path.join(precomputed_folder, method_name + '_recommendation_predictions.pkl')
                pickle.dump(predictions, open(file_path, 'wb'))
            else:
                print("\Loading recommendation model for", method_name, " ...")
                file_path = os.path.join(precomputed_folder, method_name + '_recommendation_predictions.pkl')
                predictions = pickle.load(open(file_path, 'rb'))

            recommendations = []
            for user_id, movie_id, r_ui, estimated_rating, _ in predictions:
                if movie_obj.get_movie_name(int(movie_id)) != '':
                    recommendations.append((int(movie_id), estimated_rating))
            
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            print("{:<10} {:<40} {:<10}".format("ID", "Name", "Rating"))
            for ratings in recommendations[:n]:
                print("{:<10} {:<40} {:<10.4f}".format(ratings[0], movie_obj.get_movie_name(ratings[0]), ratings[1]))
