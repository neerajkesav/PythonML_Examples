# -*- coding: utf-8 -*-
"""
DatasetLoader.
@author: neeraj
"""
import pandas

class DatasetLoader:
    """ Class DatasetLoader. To load data set. DatasetLoader have the
    following properties:
    
    Attributes:
        path: path to the data set.
        names: column names of the data set.
    
    Methods:
        __init__(): initialize variables path and names.
        load(): load data set to data from specified path and return data.
        print_shape(): print the shape of data.
    
    """
    
    path = ""
    names = []
       
    def __init__(self, path, column_names):
        """Takes arguments 'path' and 'column_names' and initializes class
        variables.        
        """
        self.path = path
        self.names = column_names
        
    def load(self):
        """Takes no arguments, load data set to 'data' from specified path
        and returns data.
        
        data: loaded with data set.       
        """
        data = pandas.read_csv(self.path, names=self.names)
        return data
        
    def print_shape(self, data):
        """Takes argument 'data', prints its shape.
        data: contains data set.         
        """
        print(data.shape)
