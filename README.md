## Python scikit-learn Machine Learning Examples

This project is created to learn Machine Learning using Python scikit-learn. This project consists of the following example:

  * Predictive Modeling on pima-indians-diabetes data set.


### Data Sets
 * pima-indians-diabetes.data - Pima Indians Diabetes Database ( https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes )


### Frameworks/Libraries
 * scikit-learn
 
  
### Getting Started

These instructions will get you a brief idea on setting up the environment and running on your local machine for development and testing purposes. 

**Prerequisities**

- python2.7
- scikit-learn
- numpy
- pandas
- matplotlib


**Setup and running tests**

1. Run `python -V` to check the installation
   
2. Install all the required libraries.
           
3. Execute the following commands from terminal to run the tests:

      `python main.py` 



###Modules/Classes
Please start exploring from module main.py

All modules in this project are listed below:

* **dataset_loader.py** - Contains class DatasetLoader to load data set. Class contains the following methods:
	
      	  `load(self)`
      	  `print_shape(self, data)`
		  
* **data_explorer.py** - Contains class DataExplorer to understand data with descriptive statistics and visualization. Class contains the following method.
	
		  `print_data_statistics(self, data)`
		  `visualize(self, data)`	

* **data_preprocessor.py** - Contains class DataPreprocessor to perform pre-processing on data set. Class contains the following methods:
	
	  	  `split_dataset(self, data, start, end, n)` 
	  	  `display_dataset(self)`
	  	  `summarize(self, data, start, end, precision)`

* **evaluator.py** - Contains class Evaluator to performs cross validation and evaluation. Class contains the following methods:
	
      	  `validate(self, model, input_set, output_set, folds, seed)`
      	  `evaluate(self, model, input_set, output_set, folds, seed, scoring)`
		  
* **model_selector.py** - Contains class ModelSelector that compares a set of models to select the best one. Class contains the following method.
	
		  `select_model(self, models, input_data, output_data, folds, seed)`

* **accuracy_improver.py** - Contains class AccuracyImprover to performs tuning and ensemble prediction. Class contains the following methods:
	
	  	  `tuning(self, model, input_set, output_set)` 
	  	  `ensemble_prediction(self, model, input_data, output_data, folds, seed)`

* **model_finalizer.py** - Contains class ModelFinalizer to finalize and save model for prediction. Class contains the following methods:
	
	  	  `split_train_test_sets(self, input_data, output_data, test_size, seed)` 
	  	  `finalize_and_save(self, model, filename, input_train, output_train)`
	  	  `predict(self, model_filename, input_test, output_test)`	
	  	  	
* **main.py** - Contains Main class to test and run the classes in this project.







