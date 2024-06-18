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
        super().__init__(experiment_df, info_csv, info_df, processing_metadata, path_to_raw_data, **kwargs)
        #drop empty columns
        if info_df is not None:
            self.info_df.drop(self.info_df.columns[self.info_df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

    def process(self,path_to_raw_data, info_csv, **kwargs):
        super().process(self,path_to_raw_data, info_csv, **kwargs)
        if 'id' not in self.info_df.columns:
            ids = []
            for i,row in self.info_df.iterrows():
                scan = row['Scan_Num']
                temp = row[TEMP]
                field = row[FIELD]
                ids.append(f'{scan}_{temp:.1f}K_{field:.1f}T')
            self.info_df['id'] = ids
        return True
