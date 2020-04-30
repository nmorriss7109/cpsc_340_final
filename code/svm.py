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
        #Control the degree of regularization:
        C = 0.4

        self.W = np.zeros((10, d))

        for e in range(maxEpochs):
            printProgressBar(self, iteration=e + 1, total=maxEpochs, prefix = 'Progress:', suffix = '   |   Epoch #: ')
            #Choose a sandom set of 50 indices
            random_indices = np.random.randint(n, size=50)

            #Take a random sample of corresponding X and y
            X_rand = Z[random_indices]
            Y_rand = Y[random_indices]

            # Loop through all tuples, (x,y), from random selection
            for (x, y) in zip(X_rand, Y_rand):
                #loop through y (10 values, one for each digit)
                for i in range(len(y)):
                    #set yi to -1 if y[i] is 0, 1 otherwise (ie. if the current digit --i-- is not represented by x)
                    if y[i] == 0:
                        yi = -1
                    else:
                        yi = 1
                    #Apply the appropriate gradient if there is error (direction determined by yi)
                    if yi*np.dot(x, self.W[i].T) <= 1:
                        self.W[i] = np.add(self.W[i], lr*C*yi*x)

        return self.W

    def predict(self,X):
        ans = []
        #loop through all test examples
        for x in X:
            #for each x, create vector to store probability that x represents the digit corresponding to the index in v
            v = np.empty([10, 1])
            #Add the bias to x
            x = np.insert(x, 0, 1)
            #loop through the weights (one weight vector per digit) and apply to x, filling in the corresponding spot in v
            for i, w in enumerate(self.W):
                v[i] = x@w
            #argmax to find the predicted digit, append that to ans
            ans.append(np.argmax(v))

        return ans
