import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

class MyCLF(BaseEstimator, ClassifierMixin):        #a child class
    def __init__(self, mode=0, learning_rate=0.01):
        self.mode = mode
        self.learning_rate = learning_rate
        self.clf = DecisionTreeClassifier(max_depth=5)
        if self.mode==1: print('MyCLF() = %s' %(self.clf))

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def score(self, X, y):
        return self.clf.score(X, y)
