from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor


class Regressor(BaseEstimator):
    
    
    def __init__(self):
        from sklearn.pipeline import Pipeline
        self.clf = Pipeline([
            ('vect', DecisionTreeRegressor()),
            ('tfidf', RandomForestRegressor()),
            ('clf', BayesianRidge(compute_score=True)),
            
        ])

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
    