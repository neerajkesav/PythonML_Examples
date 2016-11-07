# -*- coding: utf-8 -*-
"""
DataPreprocessor.
@author: neeraj
"""
import numpy

from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """Class DataPreprocessor. To perform preprocessing on data set.
    selects only relevant columns of data from data set and summarize it.
    DataPreprocessor have the following properties:
    
    Attributes:
        input: input part of the data set.
        output: output part of the data set.
    
    Methods:
        split_dataset(): splits data set to input and output.
        display_dataset(): prints 'input' and 'output' parts of 
                           data set.
        summarize(): prints the summarized data.
    """
    
    intput = []
    output = []
    
    def split_dataset(self, data, start, end, n):
        """Takes arguments 'data', 'start', 'end', 'n'.
        data: contains the data set.
        start: starting column index.
        end: ending column index.
        n: nth column.

        array: values in 'data'.       
        
        Splits the 'data'. Returns 'input' part and 'output' part of 'data'.
        """
        array = data.values
        self.input = array[:,start:end]
        self.output = array[:,n]        
        return self.input, self.output
        
         
    def display_dataset(self):
        """Takes no argument. Prints 'input' and 'output' parts 
        of data set.
        
        prints 'input' and 'output' parts of data set.
        """
        print("\nInput Data\n")         
        print(self.input)
        print("\nOutput Data\n")
        print(self.output)
    
    def summarize(self, data, start, end, precision):
        """Takes arguments 'data', 'start', 'end', 'precision'.
        data: contains the data set.
        start: starting column index.
        end: ending column index.
        precision: precision for summarizing.               
        
        prints the summarized data.
        """
        #Fitting data to scaler.
        scaler = StandardScaler().fit(data)
        rescaledX = scaler.transform(data)
        
        #Summarize transformed data
        numpy.set_printoptions(precision=precision)
        print("\nSummary\n")
        print(rescaledX[start:end,:])