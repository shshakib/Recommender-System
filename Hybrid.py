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
        for i in range(len(self.methods)):
            if isinstance(self.methods[i], surprise.KNNBasic):
                sum_scores += self.methods[i].estimate(u, i)[0] * self.weights[i]
            else:
                sum_scores += self.methods[i].estimate(u, i) * self.weights[i]
            sum_weights += self.weights[i]
            
        return sum_scores / sum_weights

    