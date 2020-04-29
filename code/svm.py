import numpy as np
from numpy.linalg import pinv
from scipy.optimize import approx_fprime
import math

from progress_bar import printProgressBar

class SVM:
    def fit(self,X,Y,maxEpochs=15000):
        # Add a bias to the X data matrix
        Z = np.concatenate((np.ones((X.shape[0],1)),X), axis = 1)
        n, d = Z.shape

        #set learning rate:
        lr = .001
        C = 0.4

        self.W = np.zeros((10, d))

        for e in range(maxEpochs):
            printProgressBar(self, iteration=e + 1, total=maxEpochs, prefix = 'Progress:', suffix = '   |   Epoch #: ')

            random_indices = np.random.randint(n, size=50)
            #Take a random sample of corresponding X and y
            X_rand = Z[random_indices]
            Y_rand = Y[random_indices]

            for (x, y) in zip(X_rand, Y_rand):
                for i in range(len(y)):
                    if y[i] == 0:
                        yi = -1
                    else:
                        yi = 1

                    if yi*np.dot(x, self.W[i].T) <= 1:
                        self.W[i] = np.add(self.W[i], lr*C*yi*x)

        return self.W

    def predict(self,X):
        ans = []
        for x in X:
            v = np.empty([10, 1])
            x = np.insert(x, 0, 1)
            for i, w in enumerate(self.W):
                v[i] = x@w

            ans.append(np.argmax(v))

        return ans
