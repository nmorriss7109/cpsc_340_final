import os
import pickle
import gzip
import argparse
import numpy as np

from sklearn.preprocessing import LabelBinarizer

from knn import KNN
from lin_reg import LeastSquares
from svm import SVM
from mlp import MLP
from cnn import CNN

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1": #K nearest neighbours
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set

        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

        def evaluate_model(model, n):
            model.fit(X,y)
            y_pred = model.predict(Xtest[:n])
            te_error = np.mean(y_pred != ytest[:n])
            print("Testing error: %.3f" % te_error)

        model = KNN(5)
        print("K-nearest neighbours")
        evaluate_model(model, n=500) #classify first n test examples

    elif question == "2": #Linear regression
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set

        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

        def evaluate_model(model):
            model.fit(X,y)
            y_pred = model.predict(Xtest)
            te_error = np.mean(y_pred != ytest)
            print("Testing error: %.3f" % te_error)

        model = LeastSquares()
        evaluate_model(model)

    elif question == "3": #Support vector machine
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set

        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

        def evaluate_model(model):
            model.fit(X,Y)
            y_pred = model.predict(Xtest)
            te_error = np.mean(y_pred != ytest)
            print("Testing error: %.3f" % te_error)

        model = SVM()
        evaluate_model(model)

    elif question == "4": #Multi layer perceptron
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set

        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

        def evaluate_model(model):
            model.fit(X,Y)
            y_pred = model.predict(Xtest)
            te_error = np.mean(y_pred != ytest)
            print("Testing error: %.3f" % te_error)

        model = MLP([200, 100])
        evaluate_model(model)

    elif question == "5": #Convolutional nerual net
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set

        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

        def evaluate_model(model):

            model.fit(X,y)
            y_pred = model.predict(Xtest[:50])
            print(y_pred)
            te_error = np.mean(y_pred != ytest[:50])
            print("Testing error: %.3f" % te_error)

        model = CNN()
        evaluate_model(model)

    else:
        print("Unknown question: %s" % question)
