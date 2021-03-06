import numpy as np

class MovingAverage:

    def __init__(self, window_size):

        self.window_size_ = window_size
        self.quadrilaterals_ = []
        self.average_ = 0
        self.weight_ = 5

    def addQuadrilateral(self, points):

        if len(self.quadrilaterals_) < self.window_size_:
            self.quadrilaterals_.append(points)

        else:
            self.quadrilaterals_.pop(0)
            self.quadrilaterals_.append(points)

    def getAverage(self):
        
        quadrilaterals = np.array(self.quadrilaterals_)
        # print(quadrilaterals.shape)
        weights = np.ones((1, self.window_size_))
        weights[0, self.window_size_-1] = self.weight_

        sum = 0
        for i in range(self.window_size_):
            sum = sum + weights[0,i] * quadrilaterals[i]

        self.average_ = sum / np.sum(weights)
        # self.average_ = np.mean(quadrilaterals, axis = 0)
        return self.average_

    def getListLength(self):
        l = len(self.quadrilaterals_)
        return l


