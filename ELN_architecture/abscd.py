'''A Class for Compiling Abs and CD Data from the J-1700.'''
from classes import *
from scipy.optimize import least_squares

#Some global variable
NM = 'NANOMETERS'
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
        if info_df is not None:
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
                        count=1
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
                            elif 'XYDATA' in line:
                                break
                            count = count+1
                    #read in csv as dataframe to correct place in info_df
                    self.info_df.at[i,'data'] = pd.read_csv(path_to_raw_data+self.info_df.at[i,'File'], skiprows=count, header=None,
                                                nrows=npts, names=np.concatenate((xlabels,ylabels)), dtype=float)
        #Use parent load function if saved data from code rather than j1700
        else:
            print('Using Parent Class Load function.')
            super().load(path_to_raw_data, data_files)

    def subtract(self, ref_id, ys, ids_to_subtract=None):
        '''Subtract off 1 spectrum's y values from (by default) all of them. 
        Include list of ids_to_subtract to only subtract ref_id off from a sub-set of the data.
        Does not check to make sure x values align!'''
        ref_idx = self.info_df.index[self.info_df['id'] == ref_id].to_list()
        if len(ref_idx) == 1:
            ref_idx = ref_idx[0]
        else:
            print('Did not select reference to subtract correctly.')
            return False
        ref_values = self.info_df.at[ref_idx, 'data'][ys]
        for i, row in self.info_df.iterrows():
            if ids_to_subtract is None or row['id'] in ids_to_subtract: #only subtract from selected samples
                self.info_df.at[i,'data'][ys] = row['data'][ys].sub(ref_values)
        #self.processing_metadata = self.processing_metadata + ' subtracted off ' + ref_id + ' from ' + str(ys)
        return True
    
    def baseline(self, ys, x_range, x_col='NANOMETERS', ignore_idx=None):
        '''Subtract off average y value of featureless region to correct for non-zero baseline'''
        for i, row in self.info_df.iterrows():
            if ignore_idx is None or i not in ignore_idx:
                ref_values = row['data'].loc[(row['data'][x_col]> x_range[0]) & (row['data'][x_col] < x_range[1])].mean(axis=0)[ys]
                self.info_df.at[i,'data'][ys] = row['data'][ys].sub(ref_values)
        #self.processing_metadata = self.processing_metadata + ' baseline subtracted off using' + x_col + str(x_range)
        return True

    def fix_changeover(self, ys, x_change, x_col='NANOMETERS', to_move='Less', ignore_idx=None, pts=2):
        '''Fix the discontinuity created by the J-1700 when the detector changes'''
        for i, row in self.info_df.iterrows():
            if ignore_idx is None or i not in ignore_idx:
                #Determine the y value before and after the discontinuity
                before = np.average(self.info_df.at[i,'data'].loc[self.info_df.at[i,'data'][x_col] < x_change].head(pts)[ys].values)
                after = np.average(self.info_df.at[i,'data'].loc[self.info_df.at[i,'data'][x_col] >= x_change].tail(pts)[ys].values)
                diff = np.subtract(after,before)
                #subtract off the difference from the appropriate side
                if to_move == 'Less' or to_move== 'less':
                    self.info_df.at[i,'data'][ys] = pd.concat((self.info_df.at[i,'data'].loc[self.info_df.at[i,'data'][x_col] < x_change][ys].add(diff),
                                                                self.info_df.at[i,'data'].loc[self.info_df.at[i,'data'][x_col] >= x_change][ys]))

                elif to_move == 'Both' or to_move == 'both':
                    self.info_df.at[i,'data'][ys] = pd.concat((self.info_df.at[i,'data'].loc[self.info_df.at[i,'data'][x_col] < x_change][ys].add(diff/pts),
                                                                self.info_df.at[i,'data'].loc[self.info_df.at[i,'data'][x_col] >= x_change][ys].sub(diff/pts)))
                elif to_move == 'More' or to_move == 'more':
                    self.info_df.at[i,'data'][ys] = pd.concat((self.info_df.at[i,'data'].loc[self.info_df.at[i,'data'][x_col] < x_change][ys],
                                                                self.info_df.at[i,'data'].loc[self.info_df.at[i,'data'][x_col] >= x_change][ys].sub(diff)))
                else:
                    print('Please specify to_move as Less, Both, or More as to_move parameter.')
                    
    def add_wavenums(self, nm_str='NANOMETERS'):
        '''Add an x value of wavenumbers by converting nanometers'''
        if nm_str in self.info_df.at[0,'data'].columns:
            for i,row in self.info_df.iterrows():
                self.info_df.at[i,'data']['Wavenums'] = np.power(row['data'][nm_str].astype(float),-1)*10000000
            print(self.info_df.at[0,'data'].columns.values)
            return True
        else:
            return False


    def add_deps(self, conc, conc_units='M', path_length=1):
        '''Add a y value of Delta Epsilon (1/M*cm) by converting from mdeg.
        conc - takes numerical value or name of dataframe column in self.data'''
        print('Path Length = ', path_length)
        
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

    def add_eps(self, conc, conc_units='M', path_length=1, abs_str='ABSORBANCE', ids=None):
        '''Add a y value of Epsilon (1/M*cm) by converting from Abs.
        conc - takes numerical value or name of dataframe column in self.data
        ids - list of ids for which rows to add eps for'''
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
        if abs_str in self.info_df.at[0,'data'].columns:
            #for each row do the math for the conversion
            for i, row in self.info_df.iterrows():
                if ids is None or row['id'] in ids:
                    # print(f'Added eps for {row['id']}.')
                    self.info_df.at[i,'data']['eps'] = np.divide(row['data'][abs_str],(concM[i] * path_length)) # type: ignore
        print(self.info_df.at[0,'data'].columns.values)

    def gauss(self, x, center, fwhm):
        '''Define the Gaussian Distribution Function'''
        width = fwhm/(2*np.sqrt(2*np.log(2)))
        return np.exp(-1/2*(x-center)**2/(width)**2)

    def resid(self, fitvars, xs, ys):
        '''Residual Calculator for Fitting to Gaussians'''
        #fit vars should be in format [energy1, energy2, ..., width1, w2, ..., scalarAbs1, sA2, ..., scalarCD1, sCD2, ...]
        num_gauss = int(len(fitvars)/(len(ys)+2))
            
        total_resid=np.array([])
        #for each y given, calculate gaussians, fit, and add to list of residuals
        for j in range(len(ys)):
            #check to see if multiple x lists
            if len(xs)==len(ys):
                x=xs[j]
            else:
                x=xs
            #get a list of the individual gaussian y values
            gauss_list = [fitvars[i+((2+j)*num_gauss)]*self.gauss(x, fitvars[i], fitvars[i+num_gauss]) for i in range(num_gauss)]
            #calculate total for Abs with current params
            total_fit = np.sum(gauss_list, axis=0)
            #calculate total residual and add to the list
            total_resid = np.concatenate((total_resid, ys[j] - total_fit))

        return total_resid

    def fit_gaussians(self, energies, fwhm, intens, id, x_col='Wavenums', y_cols=['eps','deps'], xrange=None, low_bds=None, up_bds=None, same_x=True, gtol=1e-13, ftol=1e-13, xtol=1e-13, scalar=None):
        '''A function to fit Abs/CD data to gaussian bands'''
        #make sure all guess inputs are floats 
        energies = [float(e) for e in energies]
        widths= [float(w) for w in fwhm]
        #intens = np.concatenate((intens))
        intens= [float(ints) for ints in intens]
        
        num_gauss = len(energies)
        #get x and y data
        idx = self.info_df.index[self.info_df['id']==id].to_numpy()[0]
        sample_row = self.info_df.loc[self.info_df['id']==id].copy()
        sample_row = sample_row.reset_index()
        if xrange is not None:
            row_data = sample_row.at[0,'data'].loc[(sample_row.at[0,'data'][x_col] > xrange[0]) & (sample_row.at[0,'data'][x_col] < xrange[1])]
        else:
            row_data = sample_row.at[0,'data']
        xs = row_data[x_col].to_numpy()
        ys = [row_data[y_cols].to_numpy()] if type(y_cols) is not list else row_data[y_cols].T.to_numpy()


        #normalize ys, intensities, and bounds
        areas= np.array([0]*len(ys))
        nys = []
        nintens = np.array(intens)
        #handle bounds setup
        if low_bds is not None:
            low_bds = [float(lb) for lb in low_bds]
            nl_bds = np.array(low_bds)
        else:
            nl_bds = -1*np.inf
        if up_bds is not None:
            up_bds = [float(ub) for ub in up_bds]
            nu_bds = np.array(up_bds)
        else:
            nu_bds = np.inf
        #iterate through ys, calc and store areas, and normalize
        for k in range(len(ys)):
            areas[k]=np.trapz(abs(ys[k]), x=xs if same_x else xs[k])
            if scalar is not None:
                areas[k]=areas[k]/scalar[k]
            nys.append(np.divide(ys[k], areas[k]))
            nintens[k*num_gauss:k*num_gauss+num_gauss] = np.divide(intens[k*num_gauss:k*num_gauss+num_gauss], areas[k])

            #normalize bounds
            if low_bds is not None:
                nl_bds[k*num_gauss+(2*num_gauss):k*num_gauss+(3*num_gauss)] = np.divide(low_bds[(2*num_gauss)+k*num_gauss:k*num_gauss+(3*num_gauss)], areas[k])
            if up_bds is not None:
                nu_bds[k*num_gauss+(2*num_gauss):k*num_gauss+(3*num_gauss)] = np.divide(up_bds[(2*num_gauss)+k*num_gauss:k*num_gauss+(3*num_gauss)], areas[k])
                
        #prepare input lists
        params = np.concatenate((energies, widths, nintens))
        bounds = (nl_bds, nu_bds)
        #run fit
        fit = least_squares(self.resid, params, bounds=bounds, args=(xs, nys), verbose=1, gtol=gtol, ftol=ftol, xtol=xtol)
        
        #print(fit['success'])
        #print(fit['message'])
        nresults = fit['x']
        
        results = np.array(nresults)
        for ai in range(len(areas)):
            results[ai*num_gauss+(2*num_gauss):ai*num_gauss+(3*num_gauss)] = np.multiply(nresults[ai*num_gauss+(2*num_gauss):ai*num_gauss+(3*num_gauss)],areas[ai])
        
        details = pd.DataFrame()
        details['Energy'] = results[0:num_gauss]
        details['FWHM'] = results[num_gauss:2*num_gauss]
        for i in range(len(ys)):
            ylab = 'Inten_y'+str(i)
            details[ylab] = results[(i+2)*num_gauss:(i+3)*num_gauss]
        #add Abs max value    
        #details['y0_max'] = np.multiply([gauss(0,0,w) for w in details['FWHM']],details['Inten_y0'])
        #Calc fwhm and oscillator strengths (f)
        fs = 4.61e-9*details['FWHM']*details['Inten_y0']
        #details['fwhm'] = fwhms
        details['f'] = fs
        
        # Save results to data object
        if 'fit' not in self.info_df.columns:
            self.info_df['fit'] = None
            self.info_df['fit'] = self.info_df['fit'].astype('object')
        print(f"name.info_df.at[{idx}, 'fit']")
        print(details.to_markdown())
        self.info_df.at[idx,'fit'] = results
        
        
        return results, details, fit
            

    def check_plot(self, id, result=None, x_col='Wavenums', y_cols=['eps','deps'], xrange=None, *args, **kwargs):
        '''A way to plot the fits performed by fit_gaussians
        **kwargs takes inputs for px.line'''
        if result is None:
            idx = self.info_df.index[self.info_df['id']==id].to_numpy()[0]
            fitvars = self.info_df.at[idx,'fit'] 
        else:
            fitvars= result

        #get x and y data
        sample_row = self.info_df.loc[self.info_df['id']==id].copy()
        sample_row = sample_row.reset_index()
        if xrange is not None:
            row_data = sample_row.at[0,'data'].loc[(sample_row.at[0,'data'] > xrange[0]) & (sample_row.at[0,'data'] < xrange[1])]
        else:
            row_data = sample_row.at[0,'data']
        xs = row_data[x_col].to_numpy()
        ys = [row_data[y_cols].to_numpy()] if type(y_cols) is not list else row_data[y_cols].T.to_numpy()

        #fit vars should be in format [energy1, energy2, ..., width1, w2, ..., scalarAbs1, sA2, ..., scalarCD1, sCD2, ...]
        num_gauss = int(len(fitvars)/(len(ys)+2))
        
        #create dataframe for the results
        fit = pd.DataFrame()
        fit['x'] = xs
        #for each y given, calculate gaussians, fit, and add to list of residuals
        for j in range(len(ys)):
            ylab = 'y'+str(j)
            x=xs
                
            #get a list of the individual gaussian y values
            gauss_list = [fitvars[i+((2+j)*num_gauss)]*self.gauss(x, fitvars[i], fitvars[i+num_gauss]) for i in range(num_gauss)]
            #calculate total for Abs with current params
            total_fit = np.sum(gauss_list, axis=0)
            
            #Add expt, total fit, and each gaussian
            # fit[id+'_'+ylab] = ys[j].copy()
            fit['fit_'+id+'_'+ylab]= total_fit.copy()
            for k in range(len(gauss_list)):
                lab=ylab+'_g'+str(k)
                fit[lab] = gauss_list[k]              
        fit_fig = px.line(data_frame=fit, x='x', y=[h for h in fit.columns if ylab in h])
        
        if 'fig' not in kwargs:
            fig = fit_fig
            kwargs['fig']= fig 
        else:
            kwargs['fig']= kwargs.get('fig').add_traces(fit_fig.data)
        if 'ids' not in kwargs:
            kwargs['ids'] = [id]
        fig = self.quick_plot(y=y_cols, x=x_col, **kwargs)
        return fig
        #return fit