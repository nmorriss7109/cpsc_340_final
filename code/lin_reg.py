import numpy as np
from numpy.linalg import pinv
from scipy.optimize import approx_fprime
import math

# Ordinary Least Squares
class LeastSquares:
    def fit(self,X,y):
        #self.w = solve(X.T@X, X.T@y)
        self.w = pinv(X.T@X)@X.T@y

    def predict(self, X):
        sol = X@self.w
        #return [round(x, 0) for x in sol]
        return [int(x) for x in sol]
