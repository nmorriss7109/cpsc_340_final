import numpy as np
from numpy.linalg import pinv
from scipy.optimize import approx_fprime
import math

# Ordinary Least Squares
class LeastSquares:
    def fit(self,X,y):
        n, d = X.shape
        v = np.ones((n,1))
        #add bias
        X = np.hstack((v, X))
        print(X)
        self.w = pinv(X.T@X)@X.T@y

    def predict(self, X):
        n, d = X.shape
        v = np.ones((n,1))
        #add bias
        X = np.hstack((v, X))
        sol = X@self.w
        #convert a regression to a classification (note: this doesn't really work)
        return [int(x) for x in sol]
