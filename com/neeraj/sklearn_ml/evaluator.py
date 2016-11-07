# -*- coding: utf-8 -*-
"""
Evaluator.
@author: neeraj
"""
from sklearn import cross_validation

class Evaluator:
    """Class Evaluator. Performs cross validation and evaluation on algorithm.
    Evaluator have the following methods:
    
    Methods:
        validate(): Performs validation and prints accuracy.
        evaluate(): Performs evaluation and prints mean.
        
    """
    
    def validate(self, model, input_set, output_set, folds, seed):
        """Takes arguments: 'model', 'input_set', 'output_set', 'folds', 'seed'.
        model: model to be validated.
        input_set: input part of data set.
        output_set: output part of data set.
        folds: no. of folds.
        seed: random state seed.
        
        num_instances: no. of instances in 'input_set'.
        
        Performs validation on the specified model and prints accuracy.
        """
        num_instances = len(input_set)
        #Setting kfold.
        kfold = cross_validation.KFold(n=num_instances, n_folds=folds, random_state=seed)
        #Validation.
        results = cross_validation.cross_val_score(model, input_set, output_set, cv=kfold)
        print("\nAccuracy : %.2f%% (%.2f%%)\n") % (results.mean()*100.0, results.std()*100.0)
        
    def evaluate(self, model, input_set, output_set, folds, seed, scoring):
        """Takes arguments: 'model', 'input_set', 'output_set', 'folds', 'seed', 'scoring'.
        model: model to be evaluated.
        input_set: input part of data set.
        output_set: output part of data set.
        folds: no. of folds.
        seed: random state seed.
        scoring: type of scoring.
        
        num_instances: no. of instances in 'input_set'.
        
        Performs evaluation on the specified model and prints mean.
        """        
        num_instances = len(input_set)
        #Setting kfold.
        kfold = cross_validation.KFold(n=num_instances, n_folds=folds, random_state=seed)
        #Evaluation.
        results = cross_validation.cross_val_score(model, input_set, output_set, cv=kfold, scoring=scoring)
        print("\n%s: %.3f (%.3f)\n") % (scoring, results.mean(), results.std())
        