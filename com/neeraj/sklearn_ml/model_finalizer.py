# -*- coding: utf-8 -*-
"""
ModelFinalizer.
@author: neeraj
"""
import pickle

from sklearn import cross_validation


class ModelFinalizer:
    """Class ModelFinalizer. Splits the data set to train and test sets. 
    A model is finalized and is used for prediction. ModelFinalizer 
    have the following methods:
    
        Methods:
            split_train_test_sets(): splits input and output part of data sets
                                     to train and test sets.
            finalize_and_save(): saves finalized model to disk.
            predict(): prediction using saved model.
    """
    
    def split_train_test_sets(self, input_data, output_data, test_size, seed):
        """Takes arguments: 'input_data', 'output_data', 'test_size', 'seed'.
        input_data: input part of data set.
        output_data: output part of data set.
        test_size: splitting ratio/size.
        seed: random state seed.
        
        returns input_train, input_test, output_train, output_test
        """
        return cross_validation.train_test_split(input_data, 
                                                 output_data,
                                                 test_size=test_size,
                                                 random_state=seed)

    def finalize_and_save(self, model, filename, input_train, output_train):
        """Takes arguments 'model', 'filename', 'input_train', 'output_train'.
        model: finalized model.
        filename: filename to which model to be saved.
        input_train: input part of train_set.
        output_train: output part of train_set.
        
        Saves the model to disk.
        """
        model.fit(input_train, output_train)
        #Save the model to disk
        pickle.dump(model, open(filename, 'wb' ))
        print("\nModel is saved..\n")
       
    def predict(self, model_filename, input_test, output_test):
        """Takes arguments 'model_filename', 'input_test', 'output_test'.
        model_filename: filename from which model to be loaded.
        input_test: input part of test_set.
        output_test: output part of test_set.
        
        Prints the prediction.
        """
        #Load the model from disk
        loaded_model = pickle.load(open(model_filename, 'rb' ))
        result = loaded_model.score(input_test, output_test)
        print("\nPrediction")
        print(result)
    