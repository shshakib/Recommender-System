import surprise

class Metrics:

    def MAE(predictions):
        return surprise.accuracy.mae(predictions, verbose=False)


    def RMSE(predictions):
        return surprise.accuracy.rmse(predictions, verbose=False)


    def get_top_n(predictions, n=10):
        top_n = {}
        for user_id, movie_id, r_ui, estimated_rating, _ in predictions:
            if user_id in top_n:
                top_n[user_id].append((movie_id, estimated_rating))
            else:
                top_n[user_id] = [(movie_id, estimated_rating)]
        #Sort each user's list of rated movies
        for user_id in top_n.keys():
            top_n[user_id].sort(key=lambda x: x[1], reverse=True)
        #Get top n
        for user_id in top_n.keys():
            top_n[user_id] = top_n[user_id][:n]
        
        return top_n


    def hit_rate(top_n_predicted, leave_one_predictions):
        hits = 0
        total = 0
        for user_id, movie_id, r_ui, estimated_rating, _ in leave_one_predictions:
            if user_id in top_n_predicted and movie_id in [x[0] for x in top_n_predicted[user_id]]:
                hits += 1
            total += 1
        hit_rate = hits / total if total > 0 else 0

        return hit_rate
