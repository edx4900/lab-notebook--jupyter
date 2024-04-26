'''A set of classes and functions for reading in, storing, and analyzing Solomon Lab Data.
Lab_Data serves as the parent class and each experiment and/or instrument type will have a child class.'''

#packages that will be used
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objs as go
import os

#Some Global Variables?


#Parent Class
class Lab_Data:
    '''Parent class with a variety of general functions for processing Solomon Lab Data.'''
    def __init__(self, 
                 experiment_df:'df with experiment_id, type, project (from experiment dashboard)' = None, 
                 info_csv:'file name for experiment key -- maybe remove later from class attribute' = None, 
                 info_df:'df with id and data columns (same as key)' = None, 
                 processing_metadata: 'will decide format later: maybe str' = None,
                 path_to_raw_data:'path to folder with info/data csv files' = None) -> None: # type: ignore
        self.experimen_df = experiment_df
        self.info_df = info_df
        self.processing_metadata = processing_metadata
        if path_to_raw_data is not None:
            self.process(path_to_raw_data, info_csv)

    def copy(self, child_class=Lab_Data):
        '''Copy an instance of the data Class'''
        new = child_class(experiment_df=self.experimen_df.copy(),
                       info_df=self.info_df.copy(),
                       processing_metadata= self.processing_metadata)
        return new
    
    def read(self, path_to_raw_data, info_csv):
        '''Given a folder with an info csv and data csvs, read the files, and fill in the info df without data.'''
        #parse files to find log and data
        #Collect a list of the available csv files that contain data
        data_files=[]
        for root,dirs,file in os.walk(top=path_to_raw_data):
            if root==path_to_raw_data:
                for f in file:
                    if f.endswith('.csv'):
                        data_files.append(f)
        #use pandas to read in info file
        self.info_df = pd.read_csv(info_csv)
        return data_files

    def load(self, path_to_raw_data, data_files):
        '''Convert data from raw csv files to standardized (will be instrument-specific) and put into self.info['data']'''
        #Create a column in info_df to hold the data
        self.info_df['data'] = pd.Series(dtype='object')
        #iterate through the info_df to read in the data files and store in the data column in info_df
        for i,row in self.info_df.iterrows():
            if self.info_df.at[i,'File'] in data_files:
                self.at[i,'data'] = pd.read_csv(path_to_raw_data + row['File'])
        return True

    def process(self,path_to_raw_data, info_csv):
        '''Read and Load the data to populate the class attributes.'''
        data_files = self.read(path_to_raw_data, info_csv)
        self.load(path_to_raw_data, data_files)
        return True
    
    def write(self, path_to_raw_data):
        '''Write processed data to a file.'''
        pass

    def print(self):
        '''Print information from the data object'''
        pass

    def to_md(self):
        '''Print some information and return the info dataframe in markdown for display using IPython.'''
        pass

    def quick_plot(self, x, y):
        '''Use plotly to generate a plot.'''
        pass

    def prep_plt(self):
        '''Prepare the commands to generate a matplotlib plot of the data.'''
        pass

