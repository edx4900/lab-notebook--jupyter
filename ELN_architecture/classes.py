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

import plotly.subplots

#Some Global Variables?


#Parent Class
class Lab_Data:
    '''Parent class with a variety of general functions for processing Solomon Lab Data.'''
    def __init__(self, 
                 experiment_df: 'df with experiment_id, type, project (from experiment dashboard)' = None, # type: ignore
                 info_csv:'file name for experiment key -- maybe remove later from class attribute' = None, # type: ignore
                 info_df:'df with id and data columns (same as key)' = None, # type: ignore
                 processing_metadata: 'will decide format later: maybe str' = None, # type: ignore
                 path_to_raw_data:'path to folder with info/data csv files' = None) -> None: # type: ignore
        self.experimen_df: pd.DataFrame = experiment_df
        self.info_df: pd.DataFrame = info_df
        self.processing_metadata = processing_metadata
        if path_to_raw_data is not None and info_csv is not None:
            print(f'Parsing Data from {path_to_raw_data} and {info_csv}.')
            self.process(path_to_raw_data, info_csv)

    def copy(self):
        '''Copy an instance of the data Class. Should work for child classes too.'''
        new = type(self)(experiment_df=self.experimen_df.copy(),
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
                    if f.endswith('.csv') and 'info' not in f:
                        data_files.append(f)
        #use pandas to read in info file
        self.info_df = pd.read_csv(info_csv)
        #Create a column in info_df to hold the data
        self.info_df['data'] = pd.Series(dtype='object')
        return data_files

    def load(self, path_to_raw_data, data_files):
        '''Convert data from raw csv files to standardized (will be instrument-specific) and put into self.info['data']'''
        #iterate through the info_df to read in the data files and store in the data column in info_df
        for i,row in self.info_df.iterrows():
            if self.info_df.at[i,'File'] in data_files:
                self.info_df.at[i,'data'] = pd.read_csv(path_to_raw_data + row['File'])
        return True

    def process(self,path_to_raw_data, info_csv):
        '''Read and Load the data to populate the class attributes.'''
        data_files = self.read(path_to_raw_data, info_csv)
        self.load(path_to_raw_data, data_files)
        return True

    def write(self, path_to_proc_data):
        '''Write processed data to a folder, saving the info_df and the data separately as csv files.'''
        #Fix path string if not given a terminal /
        if not path_to_proc_data.endswith('/'):
            path_to_proc_data = path_to_proc_data.strip().replace(' ','_') + '/'
        #check for and/or make directory
        if not os.path.exists(path_to_proc_data):
            os.mkdir(path_to_proc_data)
        #save each individual dataset
        for i, row in self.info_df.iterrows():
            # Manipulate the ID name to make it a reasonable filename
            name = str(row['id']).replace(' ', '_').replace('.','').replace('/','')
            #save x,y data to csv
            row['data'].to_csv(f'{path_to_proc_data}{name}.csv')
        #save the info df
        info = self.info_df.drop('data', axis=1)
        info.to_csv(f'{path_to_proc_data}info.csv')
        print(f'Data saved to {path_to_proc_data}.')
        return path_to_proc_data

    def print(self):
        '''Print information from the data object'''
        pass

    def to_md(self):
        '''Print some information and return the info dataframe in markdown for display using IPython.'''
        print('Data Columns and info_df:')
        print(self.info_df.at[0,'data'].columns.values)
        return self.info_df.drop('data',axis=1).to_markdown()

    def quick_plot(self, x= None, y= None, x_range=None, y_range=None, height=None, width=1000):
        '''Use plotly to generate a general plot.
        x and y take as imput the string for the column'''
        # Makes interactive plotly subplots from the data given specific x and y (str or list) parameters as the df column names
        # Set up y data structure, should be a list to handle plotting multiple y values stacked
        if y is None:
            y = self.info_df.at[0,'data'].columns.values[1:]
        if isinstance(y, str):
            y = [y]
        # Set up x data structure, should be the column name to use as x
        if x is None:
            x = self.info_df.at[0,'data'].columns.values[0]

        fig = plotly.subplots.make_subplots(rows=len(y), cols=1, subplot_titles=y, vertical_spacing=0.1, shared_xaxes=True)

        # Define a color sequence for lines
        #colors = px.colors.qualitative.G10 #alternative color scheme
        colors = ['blue', 'red', 'green', 'gold',  'purple', 'deepskyblue', 'orange', 'slategrey', 'brown', 'black']
        
        #Chat GPT figured out a way to link the colors and names of the plots for each y value
        for i, y_label in enumerate(y):
            for j, idx in enumerate(self.info_df.index):
                color_index = j % len(colors)  # Cycle through colors for each line in subplot
                trace_name = f'{self.info_df.at[idx, "id"]} ({y_label})'  # Include both index label and y label in trace name
                fig.add_trace(go.Scatter(x=self.info_df.at[idx, 'data'][x], y=self.info_df.at[idx, 'data'][y_label], name=trace_name, 
                                         mode='lines', line=dict(width=2, color=colors[color_index]),
                                                                 legendgroup=str(self.info_df.at[idx, "id"])), row=i+1, col=1)
            fig.update_xaxes(title_text=x, row=i+1, col=1)
            fig.update_yaxes(title_text=y_label, row=i+1, col=1)
            fig.add_hline(y=0, row=i+1, col=1) # type: ignore

            #setup default view ranges
            if y_range is not None:
                fig.update_yaxes(range=y_range[i] if len(y) > 1 else y_range, row=i+1, col=1)
            if x_range is not None:
                fig.update_xaxes(range=x_range, row=i+1, col=1)
        
        #update plot layout
        if height is None:
            height = 350*len(y) if len(y) > 1 else 400
        fig.update_layout(width=width, height=height, margin=dict(b=50, t=50, l=20)) #setup plot layout, should potentially parse title or **kwargs here too
        #fig.show()
        return fig

    def prep_plt(self):
        '''Prepare the commands to generate a matplotlib plot of the data.'''
        pass

