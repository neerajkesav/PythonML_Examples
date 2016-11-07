# -*- coding: utf-8 -*-
"""
ModelSelector.
@author: neeraj
"""
from sklearn import cross_validation

class ModelSelector:
    """Class ModelSelector. Selects the best model from a set of models.
    ModelSelector has the following method:
    
        Method:
            select_model(): selects the best model from a set of models.
                            and returns the best model.
    """
    
    def select_model(self, models, input_data, output_data, folds, seed):
        """Takes arguments: 'models', 'input_data', 'output_data', 'folds', 'seed'.
        models: array of models.
        input_data: input part of data set.
        output_data: output part of data set.
        folds: no. of folds.
        seed: random state seed.
        
        results: stores validation result of each model.
        names: name of each model.
        num_instances: no. of instances in 'input_set'.
        scoring: type of scoring.
        max_value: mean-std value of best model.
        count: count the models.
        index: index of the best model.
        
        Performs the validation of each model and returns the best model.
        """        
        results = []
        names = []
        num_instances = len(input_data)
        scoring = 'accuracy'
        max_value = 0
        count = 0
        index = 0
        
        print("\nModel Selection...\n")
        for name, model in models: 
            #Setting kfold.
            kfold = cross_validation.KFold(n=num_instances, n_folds=folds, random_state=seed)
            #Validation.
            cv_results = cross_validation.cross_val_score(model, input_data, output_data, cv=kfold, scoring=scoring)
            results.append(cv_results)            
            names.append(name)
            
            if ((cv_results.mean() - cv_results.std()) > max_value):
                max_value = cv_results.mean() - cv_results.std()
                index = count
                
            count = count + 1
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
            
        return models[index][1]     