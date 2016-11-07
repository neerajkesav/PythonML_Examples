# -*- coding: utf-8 -*-
"""
Machine Learning: Predictive Modeling on pima-indians-diabetes Dataset.
Main.
@author: neeraj
"""
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

from dataset_loader import DatasetLoader
from data_explorer import DataExplorer
from data_preprocessor import DataPreprocessor
from evaluator import Evaluator
from model_selector import ModelSelector
from accuracy_improver import AccuracyImprover
from model_finalizer import ModelFinalizer

class Main:
    """Main class to perform predictive modeling on pima-indians-diabetes 
    data set. Describes various stages in machine learning predictive 
    modeling. Consists of following method.

    Methods:
        run(): Performs various stages in predictive modeling such as:
                1. Loading Data set.
                2. Understanding Data.
                3. Data Preprocessing.
                4. Model Evaluation.
                5. Model Selection.
                6. Improving Accuracy.
                7. Finalizing Model and Prediction.
    """
    
    def run(self):
        """Performs various stages in predictive modeling"""
        #Path to Data set.
        path = "../../neeraj/resource/pima-indians-diabetes.data"
        #Column names of Data set.
        column_names = [ ' preg ' , ' plas ' , ' pres ' , ' skin ' , ' test ' , ' mass ' , ' pedi ' , ' age ' , ' class ' ]
        #Loading Data set using class DatasetLoader.
        load_data = DatasetLoader(path, column_names)
        data = load_data.load()
        load_data.print_shape(data)

        #Understanding data using class DataExplorer.
        explore_data = DataExplorer()
        explore_data.print_data_statistics(data)
        explore_data.visualize(data)

        #Performing data preprocessing.
        process_data = DataPreprocessor()
        input_set, output_set = process_data.split_dataset(data,0,8,8)
        process_data.display_dataset()
        process_data.summarize(input_set, 0, 5, 3)

        #Model evaluation using class Evaluator.
        evaluator = Evaluator()
        evaluator.validate(LogisticRegression(), input_set, output_set, 10, 7)
        evaluator.evaluate(LogisticRegression(), input_set, output_set, 10, 7,'log_loss')

        #Selecting best model using class ModelSelector.
        model = ModelSelector()
        #A set of models for selection.
        models = []
        models.append(( ' LR ' , LogisticRegression()))
        models.append(( ' LDA ' , LinearDiscriminantAnalysis()))
        models.append(( ' RF ' , RandomForestClassifier(n_estimators=100, max_features=3)))
        selected_model = model.select_model(models, input_set, output_set, 10, 7)
        print("\nSelected Model:\n %s") % (selected_model)

        #Improving Accuracy using class AccuracyImprover.
        improve_accuracy = AccuracyImprover()
        improve_accuracy.tuning(Ridge(),input_set, output_set)
        improve_accuracy.ensemble_prediction(RandomForestClassifier(n_estimators=100, max_features=3), input_set, output_set, 10, 7)

        #Finalizing the model and performing prediction.
        finalize_model = ModelFinalizer()
        input_train, input_test, output_train, output_test = finalize_model.split_train_test_sets(input_set, output_set, 0.33, 7)
        finalize_model.finalize_and_save(LogisticRegression(), "../../neeraj/resource/pima_model.sav", input_train, output_train)
        finalize_model.predict("../../neeraj/resource/pima_model.sav", input_test, output_test)

#Calling run() in Main class.        
main = Main()
main.run()        