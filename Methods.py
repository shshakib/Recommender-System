from DataSlicer import DataSlicer
from Metrics import Metrics
import numpy as np
import pickle

class Methods:

    def __init__(self, dataset, methods_dict):
        self.sliced_dataset = DataSlicer(dataset)
        self.methods_dict = methods_dict

        
    def Evaluate(self, n=10, compute = False):
        #Calculate RMSE/MAE
        metrics = {}
        result = {}
        for method_name, method_obj in self.methods_dict.items():
            print("Evaluating ", method_name, "...")
            if compute == True:
                method_obj.fit(self.sliced_dataset.get_train_dataset())
                predictions = method_obj.test(self.sliced_dataset.get_test_dataset())
                pickle.dump(predictions, open(method_name + '_predictions.pkl', 'wb'))
                #np.save('predictions.npy', predictions)
                metrics["RMSE"] = Metrics.RMSE(predictions)
                metrics["MAE"] = Metrics.MAE(predictions)

                #Leave one out
                method_obj.fit(self.sliced_dataset.get_leave_one_out_train_dataset())
                leave_one_predictions = method_obj.test(self.sliced_dataset.get_leave_one_out_test_dataset())        
                all_predictions = method_obj.test(self.sliced_dataset.get_leave_one_out_antitest_dataset())
                pickle.dump(predictions, open(method_name + '_leave_one_predictions.pkl', 'wb'))
                pickle.dump(predictions, open(method_name + '_all_predictions.pkl', 'wb'))
                #np.save('leave_one_predictions.npy', all_predictions)
                #np.save('all_predictions.npy', all_predictions)
                #Top n recommentaion for each user
                top_n_predicted = Metrics.get_top_n(all_predictions, n)
                metrics["HR"] = Metrics.hit_rate(top_n_predicted, leave_one_predictions)
            else:
                predictions = pickle.load(open(method_name + '_predictions.pkl', 'rb'))
                #predictions = np.load('predictions.npy', allow_pickle=True)
                metrics["RMSE"] = Metrics.RMSE(predictions)
                metrics["MAE"] = Metrics.MAE(predictions)
                
                all_predictions = pickle.load(open(method_name + '_all_predictions.pkl', 'rb'))
                leave_one_predictions = pickle.load(open(method_name + '_leave_one_predictions.pkl', 'rb'))
                #all_predictions = np.load('all_predictions.npy', allow_pickle=True)
                #leave_one_predictions = np.load('leave_one_predictions.npy', allow_pickle=True)
                top_n_predicted = Metrics.get_top_n(all_predictions, n)
                metrics["HR"] = Metrics.hit_rate(top_n_predicted, leave_one_predictions)


            result[method_name] = metrics.copy()
        print("{:<50} {:<10} {:<10} {:<10}".format("Methods", "RMSE", "MAE", "HR"))
        for (method, value) in result.items():
            print("{:<50} {:<10.4f} {:<10.4f} {:<10.4f}".format(method, value["RMSE"], value["MAE"], value["HR"]))

        return metrics
    
    def top_n_recommendation(self, movie_obj, test_user=85, compute = False):
        
        for method_name, method_obj in self.methods_dict.items():
            if compute == True:
                print("\nBuilding recommendation model for", method_name, " ...")
                train_dataset = self.sliced_dataset.get_full_train_dataset()
                method_obj.fit(train_dataset)
                test_dataset = self.sliced_dataset.get_user_anti_test_dataset(test_user)
                predictions = method_obj.test(test_dataset)
                pickle.dump(predictions, open(method_name + '_recommendation_predictions.pkl', 'wb'))
            else:
                print("\Loading recommendation model for", method_name, " ...")
                predictions = pickle.load(open(method_name + '_recommendation_predictions.pkl', 'rb'))

            recommendations = []
            for user_id, movie_id, r_ui, estimated_rating, _ in predictions:
                if movie_obj.get_movie_name(int(movie_id)) != '':
                    recommendations.append((int(movie_id), estimated_rating))
            
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            print("{:<10} {:<50} {:<10}".format("ID", "Name", "Rating"))
            for ratings in recommendations[:10]:
                print("{:<10} {:<50} {:<10.4f}".format(ratings[0], movie_obj.get_movie_name(ratings[0]), ratings[1]))
