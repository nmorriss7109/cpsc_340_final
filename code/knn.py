"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats

from progress_bar import printProgressBar


class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the training data
        self.y = y

    def predict(self, Xtest):
        #empty vector of appropriate size to store predictions
        y_pred = np.empty(len(Xtest))

        #iterate over Xtest, generating prediction for each
        for i in range(len(Xtest)):
            printProgressBar(self, iteration=i + 1, total=len(Xtest), prefix = 'Progress:', suffix = '   |   Example #: ')

            #array to store tuples of training examples' distance and class
            dist_class = []

            #iterate over each training example
            for j in range(len(self.X)):
                #calculate distance to Xtest example in question
                dist = np.linalg.norm(Xtest[i] - self.X[j])
                #add a new distance, label entry to dist_class
                dist_class.append([dist, self.y[j]])

            #key method to sort dist_class by distance
            def take_first(arr):
                return arr[0]

            #sort dist_class by distance
            dist_class.sort(key = take_first)
            #take the k closest examples
            k_nearest = dist_class[:self.k]
            #extract the labels
            k_nearest_labels = [x[1] for x in k_nearest]
            #find the mode of those labels
            mode_label = stats.mode(k_nearest_labels)
            #enter the calculated mode as the predicted y value
            y_pred[i] = mode_label[0]
        return y_pred
