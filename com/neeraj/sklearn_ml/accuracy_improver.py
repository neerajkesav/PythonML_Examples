# -*- coding: utf-8 -*-
"""
AccuracyImprover.
@author: neeraj
"""
import numpy

from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation

class AccuracyImprover:
    """Class AccuracyImprover. Performs tuning and ensemble prediction.
    AccuracyImprover  have the following methods:
   
       Methods:
           tuning(): Tunes using a grid search.
           ensemble_prediction(): Ensemble predictions from models.
    """
    
    def tuning(self, model, input_set, output_set):
        """Takes arguments: 'model', 'input', 'output'.
        model: model to be tuned.
        input: input part of data set.
        output: output part of data set.
        
        Performs grid search tuning.
        """
        alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])
        param_grid = dict(alpha=alphas)
        #GridSearch.
        grid = GridSearchCV(estimator=model, param_grid=param_grid)
        grid.fit(input_set, output_set)
        print(grid.best_score_)
        print(grid.best_estimator_.alpha)
    
    def ensemble_prediction(self, model, input_data, output_data, folds, seed):
        """Takes arguments: 'model', 'input_data', 'output_data', 'folds', 'seed'.
        model: model to ensemble prediction
        input_data: input part of data set.
        output_data: output part of data set.
        folds: no. of folds.
        seed: random state seed.
        
        num_instances: no. of instances in 'input_data'.
        
        ensemble prediction.
        """
        num_instances = len(input_data) 
        #Setting kfold.
        kfold = cross_validation.KFold(n=num_instances, n_folds=folds, random_state=seed)
        results = cross_validation.cross_val_score(model, input_data, output_data, cv=kfold)
        print(results.mean())