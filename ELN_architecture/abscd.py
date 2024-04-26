'''A Class for Compiling Abs and CD Data from the J-1700.'''
import classes


class AbsCD_Data(classes.Lab_Data):
    '''Abs/CD Data Class for data from the J-1700 with parent Lab_Data'''
    def __init__(self, 
                 experiment_df: 'df with experiment_id, type, project (from experiment dashboard)' = None, 
                 info_df: 'df with id and data columns (same as key)' = None, 
                 processing_metadata: 'will decide format later: maybe str' = None, 
                 path_to_raw_data: 'path to folder with info/data csv files' = None) -> None:
        super().__init__(experiment_df, info_df, processing_metadata, path_to_raw_data)


    def load(self, path_to_raw_data):
        '''Convert from raw csv files to standardized (specifically for J-1700 Abs/CD)'''
        pass