'''A Class for Compiling MCD Data From the Oxford Cryofree Magnet & the J-1700. A Child Class of the AbsCD_Data.'''
from abscd import *

#Some global variable
NM = 'NANOMETERS'
WAVE = 'Wavenums'
ABS = 'ABSORBANCE'
CD = 'CD/DC [mdeg]'
TEMP = 'SampleTemp_SetPt(K)'
FIELD = 'Field_SetPoint(T)'
TEMP_AVG = 'Avg_Sample_Temp(K)'
FIELD_AVG = 'Avg_Magnet_Field(T)'
TEMP_DEV = 'StdDev_Sample_Temp(K)'
FIELD_DEV = 'StdDev_Magnet_Field(T)'

class MCD_Data(AbsCD_Data):
    '''Abs/CD Data Class for data from the J-1700 with parent Lab_Data'''
    def __init__(self, 
                 experiment_df: 'df with experiment_id, type, project (from experiment dashboard)' = None,  # type: ignore
                 info_csv:'file name for experiment key -- maybe remove later from class attribute' = None, # type: ignore
                 info_df: 'df with id and data columns (same as key)' = None,  # type: ignore
                 processing_metadata: 'will decide format later: maybe str' = None,  # type: ignore
                 path_to_raw_data: 'path to folder with info/data csv files' = None, # type: ignore
                 **kwargs) -> None:  # type: ignore
        
        #Add dict for df column label abbreviations
        self.labels = { 'NM' : 'NANOMETERS',
                        'WAVE' : 'Wavenums',
                        'ABS' : 'ABSORBANCE',
                        'CD' : 'CD/DC [mdeg]',
                        'TEMP' : 'SampleTemp_SetPt(K)',
                        'FIELD' : 'Field_SetPoint(T)',
                        'TEMP_AVG' : 'Avg_Sample_Temp(K)',
                        'FIELD_AVG' : 'Avg_Magnet_Field(T)',
                        'TEMP_DEV' : 'StdDev_Sample_Temp(K)',
                        'FIELD_DEV' : 'StdDev_Magnet_Field(T)',
                        'SCAN_NUM' : 'Scan_Num'}

        super().__init__(experiment_df, info_csv, info_df, processing_metadata, path_to_raw_data, **kwargs)
        #drop empty columns
        if info_df is not None:
            self.info_df.drop(self.info_df.columns[self.info_df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

    def process(self, path_to_raw_data, info_csv, **kwargs):
        super().process(path_to_raw_data, info_csv, **kwargs)
        if 'id' not in self.info_df.columns:
            ids = []
            for i,row in self.info_df.iterrows():
                scan = row['Scan_Num']
                temp = row[self.labels['TEMP']]
                field = row[self.labels['FIELD']]
                ids.append(f'{scan}_{temp:.1f}K_{field:.1f}T')
            self.info_df['id'] = ids
        return True

    def update_labels(self, new_labels):
        '''Update the label abbreviations for columns with a new dict.'''
        for key in new_labels:
            self.labels[key] = new_labels[key]
        return self.labels

    def subtract(self, ref_id = -1, ys=['CD/DC [mdeg]'], ids_to_subtract=None, drop_zeros=True):
        '''Subtract off 0 T MCD spectra from the correct nonzero field spectra. 
        ref_id - dictates which zero you will be using, -1 by default is the 0T before, +1 is the 0T after, else takes id of the zero you want.
        ids_to_subtract - which spectra to subtract off of, by default is all of them'''
        #Operation to Subtract Zeroes off of data - returns a new VTVH_Data object -- NOT UPDATED YET
        
        #Get zeros
        zero_idx = self.info_df.index[self.info_df[self.labels['FIELD']] == 0].to_list()
        #Create a new column to log subtractions
        self.info_df['Subtracted'] = None

        #See which spectra to sub with each zero with builtin logic
        if ref_id in [-1, 1]:
            for zi in range(len(zero_idx)):
                if ref_id == -1: 
                    #select ones after the zero
                    if zi < len(zero_idx)-1:
                        sub_id = self.info_df.loc[(self.info_df[self.labels['FIELD']]!=0) & (self.info_df[self.labels['SCAN_NUM']] > self.info_df.at[zero_idx[zi], self.labels['SCAN_NUM']]) & (self.info_df[self.labels['SCAN_NUM']] < self.info_df.at[zero_idx[zi+1], self.labels['SCAN_NUM']])]['id'].to_numpy()
                    else:
                        sub_id = self.info_df.loc[(self.info_df[self.labels['FIELD']]!=0) & (self.info_df[self.labels['SCAN_NUM']] > self.info_df.at[zero_idx[zi], self.labels['SCAN_NUM']])]['id'].to_numpy()

                elif ref_id == 1: 
                    #select ones before the zero
                    if zi > 0:
                        sub_id = self.info_df.loc[(self.info_df[self.labels['FIELD']]!=0) & (self.info_df[self.labels['SCAN_NUM']] < self.info_df.at[zero_idx[zi], self.labels['SCAN_NUM']]) & (self.info_df[self.labels['SCAN_NUM']] > self.info_df.at[zero_idx[zi-1], self.labels['SCAN_NUM']])]['id'].values()
                    elif zi == 0:
                        sub_id = self.info_df.loc[(self.info_df[self.labels['FIELD']]!=0) & (self.info_df[self.labels['SCAN_NUM']] < self.info_df.at[zero_idx[zi], self.labels['SCAN_NUM']])]['id'].values()
                
                #Get id of the 0T for printing and then do the subtraction
                zid = self.info_df.at[zero_idx[zi], 'id']
                print(f'Subtracted {zid} off of {sub_id} for {ys}')
                super().subtract(ref_id=zid, ids_to_subtract=sub_id, ys=ys) 
                #Add label to subtracted column
                for sub_idx in self.get_idx_from_id(sub_id):
                    self.info_df.at[sub_idx, 'Subtracted'] = zid
                

            #drop the zeros from the df
            if drop_zeros:
                print('Dropped Zeros from info_df.')
                self.info_df.drop(zero_idx,axis=0, inplace=True)

        # if only want to subtract one zero off of selected
        else:
            print(f'Subtracted {ref_id} off for {ys}.')
            super().subtract(ref_id=ref_id, ids_to_subtract=ids_to_subtract, ys=ys) 

            if drop_zeros:
                print(f'Dropped {ref_id} from info_df.')
                self.info_df.drop(self.info_df.index[self.info_df['id']==ref_id] ,axis=0, inplace=True)
            for sub_idx in self.get_idx_from_id(ids_to_subtract):
                    self.info_df.at[sub_idx, 'Subtracted'] = zid
        
        #reset index so doesnt mess up other functions
        self.info_df.reset_index(inplace=True)
        
        



    def check_mirroring(self, field=None, plot=True):
        '''Do +T + -T (should be subtracted from 0 already) which should go to 0 to check for mirroring.
        Takes field as an input of either float/int or list if only want to check specific fields in the dataset.'''

        sub_data = self.copy()

        #Select the correct spectra to compare
        if field is None:
            sub_data.info_df = sub_data.info_df.loc[sub_data.info_df[self.labels['FIELD']] > 0].copy()
        elif type(field) is float or type(field) is int:
            sub_data.info_df = sub_data.info_df.loc[sub_data.info_df[self.labels['FIELD']] == field].copy()
        elif type(field) is list:
            sub_data.info_df = sub_data.info_df.loc[sub_data.info_df[self.labels['FIELD']] in field].copy()
          
        #Find negative field of same magnitude
        for i,row in sub_data.info_df.iterrows():
            oppo=self.info_df.loc[(self.info_df[self.labels['FIELD']]==(-1)*row[self.labels['FIELD']]) & (self.info_df[self.labels['TEMP']]==row[self.labels['TEMP']])].copy().reset_index().loc[0]
            sub_data.info_df.at[i,'data'][self.labels['CD']]=np.add(row['data'][self.labels['CD']],oppo['data'][self.labels['CD']])
            sub_data.info_df.at[i,'id']=row['id'] + ' + ' + oppo['id']

        
        #Plot if desired
        if plot:
            fig = sub_data.quick_plot(y=CD, x=NM)
            return fig
        else:
            return sub_data

    
    def half_subd_fields(self, field=None):
        #Do 0.5*(+T - -T) as a replacement for subtracting off zeroes - NOT UPDATED YET
        y='CD/DC [mdeg]'
        if field is None:
            subtracted = self.data.loc[self.data['FieldSet']>0].copy()
        elif type(field) is float or type(field) is int:
            subtracted = self.data.loc[self.data['FieldSet']==field].copy()
        elif type(field) is list:
            subtracted = self.data.loc[self.data['FieldSet'] in field].copy()
          
        #Find negative field of same magnitude
        for i,row in subtracted.iterrows():
            oppo=self.data.loc[(self.data['FieldSet']==(-1)*row['FieldSet']) & (self.data['TempSet']==row['TempSet'])].reset_index().loc[0]
            subtracted.at[i,y]=0.5*np.subtract(row[y],oppo[y])
            subtracted.at[i,'Id']='0.5*(' + row['Id'] + ' - ' + oppo['Id'] + ')'

        sub_data = self.copy()
        sub_data.data = subtracted
        sub_data.source = self.source + ' checked mirroring by adding opposite signed field (+T + -T) for CD only'
        return sub_data