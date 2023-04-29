import surprise

class Hybrid(surprise.AlgoBase):

    def __init__(self, methods, weights):
        surprise.AlgoBase.__init__(self)
        self.methods = methods
        self.weights = weights

    def fit(self, train_dataset):
        surprise.AlgoBase.fit(self, train_dataset)
        for method in self.methods:
            method.fit(train_dataset)
                
        return self

    def estimate(self, u, i):
        sum_scores = 0
        sum_weights = 0
        for idx in range(len(self.methods)):
            if isinstance(self.methods[idx], surprise.KNNBasic):
                sum_scores += self.methods[idx].estimate(u, i)[0] * self.weights[idx]
            else:
                sum_scores += self.methods[idx].estimate(u, i) * self.weights[idx]
            sum_weights += self.weights[idx]  
        return sum_scores / sum_weights

    