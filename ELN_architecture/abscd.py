'''A Class for Compiling Abs and CD Data from the J-1700.'''
from classes import *

#Some global variable
NM = 'NANOMETER'
WAVE = 'Wavenums'
ABS = 'ABSORBANCE'
CD = 'CD/DC [mdeg]'

class AbsCD_Data(Lab_Data):
    '''Abs/CD Data Class for data from the J-1700 with parent Lab_Data'''
    def __init__(self, 
                 experiment_df: 'df with experiment_id, type, project (from experiment dashboard)' = None,  # type: ignore
                 info_csv:'file name for experiment key -- maybe remove later from class attribute' = None, # type: ignore
                 info_df: 'df with id and data columns (same as key)' = None,  # type: ignore
                 processing_metadata: 'will decide format later: maybe str' = None,  # type: ignore
                 path_to_raw_data: 'path to folder with info/data csv files' = None, # type: ignore
                 **kwargs) -> None:  # type: ignore
        super().__init__(experiment_df, info_csv, info_df, processing_metadata, path_to_raw_data, **kwargs)
        #drop empty columns
        self.info_df.drop(self.info_df.columns[self.info_df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)


    def load(self, path_to_raw_data, data_files, j1700 = True):
        '''Convert from raw csv files to standardized (specifically for J-1700 Abs/CD).
        If data was saved using the code and not directly by the J-1700 set j1700=False to use parent function.'''
        if j1700:
            #use pandas to read in and aggregate data files
            #parse data from each J-1700 data file
            xlabels=[]
            ylabels=[]
            #go through the data files
            for i,row in self.info_df.iterrows():
                if self.info_df.at[i,'File'] in data_files:
                    #open and read each file to extract necessary infomation for reading with pandas
                    with open(path_to_raw_data+self.info_df.at[i,'File'], 'r') as f:
                        for line in f:
                            if 'UNITS' in line and 'X' in line:
                                xstr=line.split(',')[-1].replace('\n','')
                                if xstr not in xlabels:
                                    xlabels.append(xstr)
                            elif 'UNITS' in line and 'Y' in line:
                                ystr=line.split(',')[-1].replace('\n','')
                                if ystr not in ylabels:
                                    ylabels.append(ystr)
                            elif 'NPOINTS' in line:
                                npts=int(line.split(',')[-1])
                    #read in csv as dataframe to correct place in info_df
                    self.info_df.at[i,'data'] = pd.read_csv(path_to_raw_data+self.info_df.at[i,'File'], skiprows=22, header=None,
                                                nrows=npts, names=np.concatenate((xlabels,ylabels)), dtype=float)
        #Use parent load function if saved data from code rather than j1700
        else:
            print('Using Parent Class Load function.')
            super().load(path_to_raw_data, data_files)

    def subtract(self, ref_id, ys, ids_to_subtract=None):
        ref_idx = self.info_df.index[self.info_df['id'] == ref_id].to_list()
        if len(ref_idx) == 1:
            ref_idx = ref_idx[0]
        else:
            print('Did not select reference to subtract correctly.')
            return False
        ref_values = self.info_df.at[ref_idx, 'data'][ys]
        for i, row in self.info_df.iterrows():
            if ids_to_subtract is None or row['id'] in ids_to_subtract: #only subtract from selected samples
                self.info_df.at[i,'data'][ys] = row['data'][ys].sub(ref_values, axis='columns')
        #self.processing_metadata = self.processing_metadata + ' subtracted off ' + ref_id + ' from ' + str(ys)
        return True
    
    def baseline(self, ys, x_range, x_col='NANOMETERS'):
        for i, row in self.info_df.iterrows():
            ref_values = row['data'].loc[(row['data'][x_col]> x_range[0]) & (row['data'][x_col] < x_range[1])].mean(axis=0)[ys]
            self.info_df.at[i,'data'][ys] = row['data'][ys].sub(ref_values, axis='columns')
        #self.processing_metadata = self.processing_metadata + ' baseline subtracted off using' + x_col + str(x_range)
        return True
    

    def add_wavenums(self):
    #Add an x value of wavenumbers by converting nanometers
        if 'NANOMETERS' in self.info_df.at[0,'data'].columns:
            for i,row in self.info_df.iterrows():
                self.info_df.at[i,'data']['Wavenums'] = np.power(row['data']['NANOMETERS'],-1)*10000000
            print(self.info_df.at[0,'data'].columns.values)
            return True
        else:
            return False


    def add_deps(self, conc, conc_units='M', path_length=1):
        '''Add a y value of Delta Epsilon (1/M*cm) by converting from mdeg.
        conc - takes numerical value or name of dataframe column in self.data'''
        
        #Handle concentration input type
        if type(conc) is str:
            conc = self.info_df[conc].to_numpy()
        else:
            try:
                conc = np.ones(len(self.info_df))*float(conc)
            except:
                print('Please give conc as a number or df column name.')
                return False

        #Convert to M
        if conc_units == 'mM':
            concM = conc / 1000  # Convert mM to M
        elif conc_units == 'uM':
            concM = conc / 1000000  # Convert µM to M
        elif conc_units == 'M':
            concM = conc

        #check for units to convert from
        if 'CD/DC [mdeg]' in self.info_df.at[0,'data'].columns:
            #for each row do the math for the conversion
            for i, row in self.info_df.iterrows():
                self.info_df.at[i,'data']['deps'] = np.divide(row['data']['CD/DC [mdeg]'],(concM[i] * path_length * 32980)) # type: ignore
        print(self.info_df.at[0,'data'].columns.values)

    def add_eps(self, conc, conc_units='M', path_length=1):
        '''Add a y value of Epsilon (1/M*cm) by converting from Abs.
        conc - takes numerical value or name of dataframe column in self.data'''
        #Handle concentration input type
        if isinstance(conc, str):
            conc = self.info_df[conc].to_numpy()
        else:
            try:
                conc = np.ones(len(self.info_df))*float(conc)
            except ValueError:
                print('Please give conc as a number or df column name.')
                return False

        # Convert to M
        if conc_units == 'mM':
            concM = conc / 1000  # Convert mM to M
        elif conc_units == 'uM':
            concM = conc / 1000000  # Convert µM to M
        elif conc_units == 'M':
            concM = conc
        
        #check for units to convert from
        if 'ABSORBANCE' in self.info_df.at[0,'data'].columns:
            #for each row do the math for the conversion
            for i, row in self.info_df.iterrows():
                self.info_df.at[i,'data']['eps'] = np.divide(row['data']['ABSORBANCE'],(concM[i] * path_length)) # type: ignore
        print(self.info_df.at[0,'data'].columns.values)