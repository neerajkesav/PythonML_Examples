# -*- coding: utf-8 -*-
"""
DatasetExplorer.
@author: neeraj
"""
import matplotlib.pyplot as plot

from pandas.tools.plotting import scatter_matrix

class DataExplorer:
    """Class DataExplorer. To understand data with descriptive statistics 
    and visualization. DatasetExplorer have the following properties:
    
    Methods:
        print_data_statistics():prints the descriptive statistics 
                                of 'data'.
        visualize(): plots the scatter matrix of 'data'.
    """    
    
    def print_data_statistics(self, data):
        """Takes argument 'data' and prints its descriptive statistics.
        data: contains the data set.
        
        data_description: descriptive statistics of 'data'.
        
        prints descriptive statistics.
        """
        data_description = data.describe()
        print(data_description)
    
    def visualize(self, data):
        """Takes argument 'data' and plots the scatter matrix.
        data: contains the data set.
        
        prints scatter plot.
        """
        scatter_matrix(data)
        plot.show()