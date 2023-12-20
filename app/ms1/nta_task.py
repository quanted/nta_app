import pandas as pd
import os
import csv
import time
import logging
import traceback
import shutil
import json
from datetime import datetime
from dask.distributed import Client, LocalCluster, fire_and_forget

from . import functions_Universal_v3 as fn
from . import toxpi
from .utilities import *  #connect_to_mongoDB, connect_to_mongo_gridfs, reduced_file, api_search_masses, api_search_formulas,

from . import task_functions as task_fun
from . WebApp_plotter import WebApp_plotter
import io
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# import seaborn as sns
try:
    import seaborn as sns
except ModuleNotFoundError:
    print("Seaborn is not installed. Please run 'pip install seaborn' to install it.")


#os.environ['IN_DOCKER'] = "False" #for local dev - also see similar switch in tools/output_access.py
NO_DASK = False  # set this to true to run locally without test (for debug purposes)

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("nta_app.ms1")
logger.setLevel(logging.INFO)

def run_nta_dask(parameters, input_dfs, tracer_df = None, run_sequence_pos_df = None, run_sequence_neg_df = None, jobid = "00000000", verbose = True):
    in_docker = os.environ.get("IN_DOCKER") != "False"
    mongo_address = os.environ.get('MONGO_SERVER')
    
    if NO_DASK:
        run_nta(parameters, input_dfs, tracer_df, run_sequence_pos_df, run_sequence_neg_df, mongo_address, jobid, verbose, in_docker = in_docker)
        return
    if not in_docker:
        logger.info("Running in local development mode.")
        logger.info("Detected OS is {}".format(os.environ.get("SYSTEM_NAME")))
        local_cluster = LocalCluster(processes=False)
        dask_client = Client(local_cluster)
    else:
        dask_scheduler = os.environ.get("DASK_SCHEDULER")
        logger.info("Running in docker environment. Dask Scheduler: {}".format(dask_scheduler))
        dask_client = Client(dask_scheduler)

    input_dfs_size= len(input_dfs)
    logger.info("Before scatter input_dfs_size: {}".format(input_dfs_size))
    dask_input_dfs = dask_client.scatter(input_dfs)
    dask_input_dfs_size= len(dask_input_dfs)
    logger.info("After scatter dask_input_dfs_size: {}".format(dask_input_dfs_size))

    logger.info("Submitting Nta Dask task")
    task = dask_client.submit(run_nta, parameters, dask_input_dfs, tracer_df, run_sequence_pos_df, run_sequence_neg_df, mongo_address, jobid, verbose,
                              in_docker = in_docker)
    fire_and_forget(task)


def run_nta(parameters, input_dfs, tracer_df = None, run_sequence_pos_df = None, run_sequence_neg_df = None, mongo_address = None, jobid = "00000000", verbose = True,
            in_docker = True):
    nta_run = NtaRun(parameters, input_dfs, tracer_df, run_sequence_pos_df, run_sequence_neg_df, mongo_address, jobid, verbose, in_docker = in_docker)
    try:
        nta_run.execute()
    except Exception as e:
        trace = traceback.format_exc()
        logger.info(trace)
        fail_step = nta_run.get_step()
        nta_run.set_status("Failed on step: " + fail_step)
        error = repr(e)
        nta_run.set_except_message(error)
        raise e
    return True


class NtaRun:
    
    def __init__(self, parameters=None, input_dfs=None, tracer_df=None, run_sequence_pos_df = None, run_sequence_neg_df = None, mongo_address = None, jobid = "00000000",
                 verbose = True, in_docker = True):
        logger.info("Initializing NtaRun Task")
        logger.info("parameters= {}".format(parameters))
        self.parameters = parameters
        self.tracer_df = tracer_df
        self.tracer_dfs_out = None
        self.run_sequence_pos_df = run_sequence_pos_df
        self.run_sequence_neg_df = run_sequence_neg_df
        self.dfs = input_dfs
        self.dfs_flagged = None # DFs that will retain occurrences failing CV values
        self.dupes = None
        self.docs = None
        self.doc_combined = None
        self.df_combined = None
        self.df_flagged_combined = None
        self.pass_through = None
        self.mpp_ready = None
        self.mpp_ready_flagged = None
        self.search_results = None
        self.search = None
        self.jobid = jobid
        self.verbose = verbose
        self.in_docker = in_docker
        self.mongo_address = mongo_address
        self.mongo = connect_to_mongoDB(self.mongo_address)
        self.gridfs = connect_to_mongo_gridfs(self.mongo_address)
        self.base_dir = os.path.abspath(os.path.join(os.path.abspath(__file__),"../../.."))
        self.data_map = {}
        self.tracer_map = {}
        self.occurrence_heatmap_map = {}
        self.cv_scatterplot_map = {}
        #self.data_dir = os.path.join(self.base_dir, 'data', self.jobid)
        #self.new_download_dir = os.path.join(self.data_dir, "new")
        self.step = "Started"  # tracks the current step (for fail messages)
        #os.mkdir(self.data_dir)
        #os.mkdir(self.new_download_dir)
        self.tracer_plots_out = []
        self.occurrence_heatmaps_out = []
        self.cv_scatterplots_out = []


    def execute(self):

        self.step = "Check for existence of required columns"
        # 0: check existence of "Ionization mode" column
        self.check_existence_of_ionization_mode_column(self.dfs)  
        # 0: check existence of 'mass column'
        self.check_existence_of_mass_column(self.dfs)
        # 0: check for alternate spellings of 'Retention_Time' column
        self.check_retention_time_column(self.dfs)
        
        # 0: sort dataframe columns alphabetically
        self.dfs = [df.reindex(sorted(df.columns), axis=1) if df is not None else None for df in self.dfs]
        # 0: create a status in mongo
        self.set_status('Processing', create = True)
        # 0: create an analysis_parameters sheet
        self.create_analysis_parameters_sheet()
        # 1: drop duplicates and throw out void volume
        self.step = "Dropping duplicates"
        self.assign_id()
        self.pass_through_cols()
        self.filter_void_volume(float(self.parameters['minimum_rt'][1])) # throw out features below this (void volume)
        self.filter_duplicates()
        if self.verbose:
            logger.info("Dropped duplicates.")
            logger.info("dfs.size(): {}".format(len(self.dfs)))
            if self.dfs[0] is not None:
                logger.info("POS df length: {}".format(len(self.dfs[0])))
            if self.dfs[1] is not None:
                logger.info("NEG df length: {}".format(len(self.dfs[1])))
            #print(self.dfs[0])
        
        logger.info("self.dfs[1] pre-stat columns= {}".format(self.dfs[1].columns))    
        
        # 2: statistics
        self.step = "Calculating statistics"
        self.calc_statistics()
        if self.verbose:
            logger.info("Calculated statistics.")
            if self.dfs[0] is not None:
                logger.info("POS df length: {}".format(len(self.dfs[0])))
            if self.dfs[1] is not None:
                logger.info("NEG df length: {}".format(len(self.dfs[1])))
            #print(self.dfs[0])
            #print(str(list(self.dfs[0])))

        #logger.info("self.dfs.size(): {}".format(len(self.dfs)))
        #for df in self.dfs:
        #    if df is not None:
        #        logger.info("df = {}".format(df.to_string()))
        # explanation of the following line:
        # . self.dfs is a list of dataframes
        # . the if statement is checking if the dataframe is not None
        # . if the dataframe is not None, then the dataframe is passed to the function cal_detection_count
        # . the result of the function is stored in the list
        # . the list is assigned to self.dfs
        
        # 10/30/23 Calculating detection counts now happens in the 'Clean Features' step -- TMF
        #self.dfs = [task_fun.cal_detection_count(df) if df is not None else None for df in self.dfs]
 

        # 2.1: Occurrence heatmap
        self.step = "Create heatmap"
        self.occurrence_heatmap(self.dfs)  


        # 3: check tracers (optional)
        self.step = "Checking tracers"
        self.check_tracers()
        if self.verbose:
            logger.info("Checked tracers.")
            #print(self.tracer_dfs_out)
        # counting occrrences of each feature after cleaning
        
        # 3.1: CV Scatterplog
        self.step = "Create scatterplot"
        self.cv_scatterplot(self.dfs)  

        # 4: clean features
        self.step = "Cleaning features"
        self.clean_features()
        
        if self.verbose:
            logger.info("Cleaned features.")
            if self.dfs[0] is not None:
                logger.info("POS df length: {}".format(len(self.dfs[0])))
            if self.dfs[1] is not None:
                logger.info("NEG df length: {}".format(len(self.dfs[1])))
            #print(self.dfs[0])

        # 4.1: Merge detection count columns onto tracers for export
        self.step = "Merge detection counts onto tracers"
        self.merge_columns_onto_tracers()

        # commented out for NTAW-94
        # # 5: create flags
        self.step = "Creating flags"
        # self.create_flags()
        if self.verbose:
            logger.info("Created flags.")
            if self.dfs[0] is not None:
                logger.info("POS df length: {}".format(len(self.dfs[0])))
                # NTAW-94 set flag for dashboard search
                self.dfs[0]['For_Dashboard_Search'] = '1'
            if self.dfs[1] is not None:
                logger.info("NEG df length: {}".format(len(self.dfs[1])))
                # NTAW-94 set flag for dashboard search
                self.dfs[1]['For_Dashboard_Search'] = '1'
            #print(self.dfs[0])
        # hardcoded to search all features against dsstox

        # 6: combine modes
        self.step = "Combining modes"
        self.combine_modes()
        if self.verbose:
            logger.info("Combined modes.")
            logger.info("combined df length: {}".format(len(self.df_combined)))
            #print(self.df_combined)

        # 7: search dashboard
        if self.parameters['search_dsstox'][1] == 'yes':
            self.step = "Searching dsstox database"
            self.perform_dashboard_search()
            if self.parameters['search_hcd'][1] == 'yes':
                self.perform_hcd_search()
            # NTAW-94 comment out the following line. toxpi is no longer being used
            # self.process_toxpi()
        if self.verbose:
            logger.info("Final result processed.")
        #if self.verbose:
        #    logger.info("Download files removed, processing complete.")
        
        # 8: Store data to MongoDB
        self.step = "Storing data"
        self.store_data()

        # 9: set status to completed
        self.step = "Displaying results"
        self.set_status('Completed')


    def check_existence_of_ionization_mode_column(self, input_dfs):
        """
        Check and ensure the existence of the 'Ionization_Mode' column in a list of DataFrames.

        This function iterates through a list of DataFrames, typically representing positive and negative ionization modes,
        and checks if the 'Ionization_Mode' column is present. If not found, it adds the column to the DataFrame with
        predefined values based on the mode.

        Args:
            self: The instance of the class (typically associated with object-oriented programming).
            input_dfs (list of pandas.DataFrame): A list of pandas DataFrames to check and modify.

        Returns:
            None: This function operates in place and modifies the input DataFrames.

        Example:
            input_dfs = [positive_mode_df, negative_mode_df]
            checker = IonizationModeChecker()
            checker.check_existence_of_ionization_mode_column(input_dfs)
            # The 'Ionization_Mode' column will be added to DataFrames if missing, with values 'Esi+' for positive mode
            # and 'Esi-' for negative mode.
        """        
        # the zeroth element of input_dfs is the positive mode dataframe
        ionizationMode = "Esi+"
        for df in input_dfs:
            if df is not None:
                if 'Ionization_Mode' not in df.columns:
                    # create a new column with the header of "Ionization_Mode" and values of ionizationMode
                    df['Ionization_Mode'] = ionizationMode
            # the first element of input_dfs is the negative mode dataframe
            ionizationMode = "Esi-"
        return


    def check_existence_of_mass_column(self, input_dfs):
        """
        Check the existence of a 'Mass' or 'm/z' column in input dataframes and handle it accordingly.

        Args:
            input_dfs (list of pandas DataFrames): A list of dataframes to check.

        This function checks each dataframe in the input list for the presence of a 'Mass' or 'm/z' column. If either of these columns is found, it takes appropriate action based on the ionization mode. If neither column exists, it raises a ValueError.

        Note:
            - If 'Mass' already exists, no action is taken.
            - If 'm/z' exists, a 'Mass' column is created by subtracting 1.0073 for "Esi+" mode and adding 1.0073 for "Esi-" mode.

        Raises:
            ValueError: If neither 'Mass' nor 'm/z' columns are present in the input dataframe.

        Returns:
            None
        """
        ionizationMode = "Esi+"

        # Iterate through two dataframes (assumes there are two in the input)
        for i in range(0, 2):
            df = input_dfs[i]

            # If it's the second iteration, change the ionization mode to "Esi-"
            if i == 1:
                ionizationMode = "Esi-"

            if df is not None:
                if 'Mass' in df.columns:
                    pass  # 'Mass' column already exists, no action needed
                elif 'm/z' in df.columns:
                    if ionizationMode == "Esi+":
                        df['Mass'] = df['m/z'] - 1.0073
                    elif ionizationMode == "Esi-":
                        df['Mass'] = df['m/z'] + 1.0073
                else:
                    raise ValueError("Either Mass or m/z column must be in the input file. (Check spelling!)")

        return
    
    
    def check_retention_time_column(self, input_dfs):
        
        # Check for the existence of alternate spellings of 'Retention_Time' column in input dataframes and rename to "Retention_Time".
        
        for df in input_dfs:
            if df is not None:
                # Check to see if there is not the expected spelling of "Retention_Time" column
                #if 'Retention_Time' not in df.columns:
                # replace alternative capitalizations
                df.rename(columns={'Retention_time': 'Retention_Time', 'RETENTION_TIME': 'Retention_Time'}, inplace=True)
                # replace rt/RT
                df.rename(columns={'rt': 'Retention_Time', 'RT': 'Retention_Time'}, inplace=True)
                # replace "Ret. Time" (SCIEX data)
                df.rename(columns={'Ret._Time': 'Retention_Time'}, inplace=True)
                if 'Retention_Time' not in df.columns:
                    raise ValueError("Retention_Time column must be in the input file. (Check spelling!)")

        return


    def create_analysis_parameters_sheet(self):
        # logger.info("create_analysis_parameters_sheet: inputParameters: {} ".format(inputParameters))

        # create a dataframe to store analysis parameters
        columns = ['Parameter', 'Value']
        df_analysis_parameters = pd.DataFrame(columns=columns)

        # loop through keys in self.parameters and log them
        for key in self.parameters:
            logger.info("key: {}".format(key))
            label = self.parameters[key][0]
            value = self.parameters[key][1]
            df_analysis_parameters.loc[len(df_analysis_parameters)] = [label, value]

        # add the dataframe to the data_map with the sheet name of 'Analysis_Parameters'
        self.data_map['Analysis_Parameters'] = df_analysis_parameters

        return


    def set_status(self, status, create = False):
        posts = self.mongo.posts
        time_stamp = datetime.utcnow()
        post_id = self.jobid + "_" + "status"
        if create:
            posts.update_one({'_id': post_id},{'$set': {'_id': post_id,
                                                   'date': time_stamp,
                                                   'status': status,
                                                   'error_info': ''}},
                             upsert=True)
        else:
            posts.update_one({'_id': post_id}, {'$set': {'_id': post_id,
                                                    'status': status}},
                             upsert=True)


    def set_except_message(self, e):
        posts = self.mongo.posts
        time_stamp = datetime.utcnow()
        post_id = self.jobid + "_" + "status"
        posts.update_one({'_id': post_id},{'$set': {'_id': post_id,
                                              'date': time_stamp,
                                              'error_info': e}},
                         upsert=True)


    def get_step(self):
        return self.step
    
    
    def assign_id(self):
        if self.dfs[0] is not None and self.dfs[1] is not None:
            self.dfs[0] = task_fun.assign_feature_id(self.dfs[0])
            self.dfs[1] = task_fun.assign_feature_id(self.dfs[1], start=len(self.dfs[0].index)+1)
        elif self.dfs[0] is not None:
            self.dfs[0] = task_fun.assign_feature_id(self.dfs[0])
        else:
            self.dfs[1] = task_fun.assign_feature_id(self.dfs[1])
        return
   
    
    def pass_through_cols(self):
        self.pass_through = [task_fun.passthrucol(df)[0] if df is not None else None for df in self.dfs]
        self.dfs = [task_fun.passthrucol(df)[1] if df is not None else None for df in self.dfs]
        return
   
    
    def filter_void_volume(self, min_rt):
        self.dfs = [df.loc[df['Retention_Time'] > min_rt].copy() if df is not None else None for index, df in enumerate(self.dfs)]
        return
  
    
    def filter_duplicates(self):
        #self.dfs = [task_fun.duplicates(df) if df is not None else None for df in self.dfs] # Deprecated 10/30/23 -- TMF
        #self.dfs, self.dupes = map(list, zip(*[task_fun.duplicates(df) if df is not None else None for df in self.dfs]))
        self.dupes = [task_fun.duplicates(df)[1] if df is not None else None for df in self.dfs]
        self.dfs = [task_fun.duplicates(df)[0] if df is not None else None for df in self.dfs]
        return


    def calc_statistics(self):
        ppm = self.parameters['mass_accuracy_units'][1]== 'ppm'
        
        self.dfs = [task_fun.chunk_stats(df) if df is not None else None for df in self.dfs]
        self.dupes = [task_fun.chunk_stats(df) if df is not None else None for df in self.dupes]
        
        if self.dfs[0] is not None and self.dfs[1] is not None:
            mass_accuracy = float(self.parameters['mass_accuracy'][1])
            rt_accuracy = float(self.parameters['rt_accuracy'][1])
            self.dfs[0] = task_fun.adduct_identifier(self.dfs[0], mass_accuracy, rt_accuracy, ppm,
                                                 ionization='positive', id_start=1)
            self.dfs[1] = task_fun.adduct_identifier(self.dfs[1], mass_accuracy, rt_accuracy, ppm,
                                                 ionization='negative', id_start=len(self.dfs[0].index)+1)
            self.data_map['Feature_statistics_positive'] = task_fun.column_sort_DFS(pd.merge(self.dfs[0], self.pass_through[0], how='left', on=['Feature_ID']))
            self.data_map['Feature_statistics_negative'] = task_fun.column_sort_DFS(pd.merge(self.dfs[1], self.pass_through[1], how='left', on=['Feature_ID']))
        elif self.dfs[0] is not None:
            mass_accuracy = float(self.parameters['mass_accuracy'][1])
            rt_accuracy = float(self.parameters['rt_accuracy'][1])
            self.dfs[0] = task_fun.adduct_identifier(self.dfs[0], mass_accuracy, rt_accuracy, ppm,
                                                 ionization='positive', id_start=1)
            self.data_map['Feature_statistics_positive'] = task_fun.column_sort_DFS(pd.merge(self.dfs[0], self.pass_through[0], how='left', on=['Feature_ID']))
        else:
            mass_accuracy = float(self.parameters['mass_accuracy'][1])
            rt_accuracy = float(self.parameters['rt_accuracy'][1])
            self.dfs[1] = task_fun.adduct_identifier(self.dfs[1], mass_accuracy, rt_accuracy, ppm,
                                                 ionization='negative', id_start=1)
            self.data_map['Feature_statistics_negative'] = task_fun.column_sort_DFS(pd.merge(self.dfs[1], self.pass_through[1], how='left', on=['Feature_ID']))
        return


    def cv_scatterplot(self, input_dfs):

        plt.rcdefaults()

        titleText = "CV vs. Abundance"
        
        max_replicate_cv_value = self.parameters['max_replicate_cv'][1]
        # convert to float
        max_replicate_cv_value = float(max_replicate_cv_value)

        # get dataframe 'Feature_statistics_positive' if it exists else None
        dfPos = self.data_map['Feature_statistics_positive'] if 'Feature_statistics_positive' in self.data_map else None

        # get dataframe 'Feature_statistics_negative' if it exists else None  
        dfNeg = self.data_map['Feature_statistics_negative'] if 'Feature_statistics_negative' in self.data_map else None

        dfTracer = self.data_map['Tracer_Sample_Results'] if 'Tracer_Sample_Results' in self.data_map else None
        
        # Add conditional for if tracers exist:
        if dfTracer is not None:
            tracers = dfTracer[['Observed_Mass', 'Observed_Retention_Time']].copy()
            tracers.rename({'Observed_Mass':'Mass'}, axis=1, inplace=True)
            tracers.rename({'Observed_Retention_Time':'Retention_Time'}, axis=1, inplace=True)
            tracers['spike'] = 1

        # combine the two dataframes. Ignnore non-existing dataframes
        dfCombined = pd.concat([dfPos, dfNeg], axis=0, ignore_index=True, sort=False) if dfPos is not None and dfNeg is not None else dfPos if dfPos is not None else dfNeg if dfNeg is not None else None

        # # log the dataframes if not None
        # if dfPos is not None:
        #     logger.info("dfPos= {}".format(dfPos.columns.values))
        # if dfNeg is not None:
        #     logger.info("dfNeg= {}".format(dfNeg.columns.values))
        # if dfCombined is not None:
        #     logger.info("dfCombined= {}".format(dfCombined.columns.values))
        
        
        # Get sample headers                
        all_headers = task_fun.parse_headers(dfCombined) 
        #logger.info("all_headers= {}".format(all_headers))
        sam_headers = [sublist[0][:-1] for sublist in all_headers if len(sublist) > 1]
        #logger.info("sam_headers= {}".format(sam_headers))

        # Isolate sample_groups from stats columns
        prefixes = ['Mean_','Median_', 'CV_', 'STD_', 'N_Abun_', 'Replicate_Percent_']
        sample_groups = [item for item in sam_headers if not any(x in item for x in prefixes)]
        #logger.info("sample_groups= {}".format(sample_groups))

        # Blank_MDL - need to check what the blank samples are actually named
        blank_strings = ['MB', 'Mb', 'mb', 'BLANK', 'Blank', 'blank', 'BLK', 'Blk']
        blank_col = [item for item in sample_groups if any(x in item for x in blank_strings)]
        #logger.info("blank_col= {}".format(blank_col))

        blank_mean = 'Mean_' + blank_col[0]
        #logger.info("blank_mean= {}".format(blank_mean))
        blank_std = 'STD_' + blank_col[0]
        #logger.info("blank_std= {}".format(blank_std))

        dfCombined['MDL'] = dfCombined[blank_mean] + 3*dfCombined[blank_std]
        #logger.info("dfCombined['MDL']= {}".format(dfCombined['MDL']))
        #logger.info("dfCombined= {}".format(dfCombined.columns.values))

        # Find CV cols from df
        cv_cols = ['CV_' + col for col in sample_groups]
        #logger.info("cv_cols= {}".format(cv_cols))
        rper_cols = ['Replicate_Percent_' + col for col in sample_groups]
        #logger.info("rper_cols= {}".format(rper_cols))
        med_cols = ['Median_' + col for col in sample_groups]
        #logger.info("med_cols= {}".format(med_cols))

        # Grab CV cols from df
        cv_df = dfCombined[cv_cols]
        #logger.info("cv_df= {}".format(cv_df.columns.values))
        rper_df = dfCombined[rper_cols]
        #logger.info("rper_df= {}".format(rper_df.columns.values))
        med_df = dfCombined[med_cols]
        #logger.info("med_df= {}".format(med_df.columns.values))

        # Blank out cvs in samples with <2 samples -- NEED TO UPDATE TO REPLICATE PERCENT
        for x,y,z in zip(cv_cols, rper_cols, med_cols):
            # Replace cv_df values with nan in cv_col for n_abun and MDL cutoffs
            cv_df.loc[dfCombined[y]<0.67, x] = np.nan
            cv_df.loc[dfCombined[z]<=dfCombined['MDL'], x] = np.nan
        #logger.info("#1 cv_df= {}".format(cv_df.values))

        # Find CV cols from df
        cv_cols = ['CV_' + col for col in sample_groups]

        # Grab CV cols from df
        cv_df = dfCombined[cv_cols]
        cv_df['Mass'] = dfCombined['Mass']
        cv_df['Retention_Time'] = dfCombined['Retention_Time']

        # Find mean cols from df
        mean_cols = ['Mean_' + col for col in sample_groups]

        # Grab mean cols from df
        mean_df = dfCombined[mean_cols]

        li=[]
        blanks=['MB1','BLK', 'Blank', 'BLANK', 'blank', 'MB', 'mb']
        # stds=['Standard','Pooled', 'SSC', 'BSD', 'BS2', 'MSD', 'MS']

        # Take each sample's CV and mean, store in dummy variable
        # Add sample type, and whether or not native samples are present or not
        # Append to list
        for x in sample_groups:
            cv = 'CV_'+x
            mean = 'Mean_'+x

            dum = pd.concat([cv_df[cv], mean_df[mean]], axis=1)
            dum.rename({cv:'CV'}, axis=1, inplace=True)
            dum.rename({mean:'Mean'}, axis=1, inplace=True)

            dum['sample'] = x
            dum['Mass'] = cv_df['Mass']
            dum['Retention_Time'] = cv_df['Retention_Time']
            
            if any(i in x for i in blanks):
                dum['type'] = 'blank'
                # dum['native'] = 0
            # elif any(i in x for i in stds):
            #     dum['type'] = 'spiked'
            #     dum['native'] = 1
            else:
                dum['type'] = 'sample'
                # dum['native'] = 0
            
            li.append(dum)

        # Concatenate plot, drop NAs
        plot = pd.concat(li)
        plot.dropna(axis=0, subset=['CV', 'Mean'], how='any', inplace=True)
        
        # Conditional for if tracers are present:
        if dfTracer is not None:
            # Merge df with tracers to get labels
            plot2 = pd.merge(plot, tracers, how='left', on=['Mass', 'Retention_Time'])
            # plot2 = pd.merge(plot2, utracers, how='left', on=['Mass', 'Retention_Time'])
        else:
            # If tracer plot doesn't exist, still need to create a spike column that is empty
            plot['spike'] = ''
            plot2 = plot.copy()
        
        plot2.replace(np.nan, 0, inplace=True)

        f, axes = plt.subplots(1,2)

        f.set_figheight(5)
        f.set_figwidth(15)

        palette = ['whitesmoke', 'firebrick']
        sns.set_palette(palette, 2)

        a=sns.scatterplot(data=plot2.loc[((plot2['type']=='blank')),:].sort_values('spike'),
                        x='Mean', y='CV', hue = 'spike',
                        edgecolor = 'black', alpha = 0.5, ax = axes[0])

        # a.axhline(y=1.25, color='red', linestyle="dashed", linewidth=1.5, alpha=1)
        # a.text(1000000000, 1.4, 'CV = 1.25', ha='center', va='center_baseline', weight='bold', size = 12)
        a.axhline(y=max_replicate_cv_value, color='red', linestyle="dashed", linewidth=1.5, alpha=1)
        a.text(1000000000, max_replicate_cv_value+0.1, 'CV = {}'.format(max_replicate_cv_value), ha='center', va='center_baseline', weight='bold', size = 12)
        '''
        sns.scatterplot(data=plot2.loc[((plot2['type']=='blank')&(plot2['spike']==1)),:],
                        x='Mean', y='CV', color="firebrick",
                        edgecolor = 'black', alpha = 0.15, ax = axes[0])
        '''
        legend = a.legend(title = "Features")
        legend.get_texts()[0].set_text('unknowns')
        legend.get_texts()[1].set_text('ISTDs')

        # frame = legend.get_frame() #sets up for color, edge, and transparency
        # frame.set_facecolor('lightgray') #color of legend
        # frame.set_edgecolor('black') #edge color of legend
        # frame.set_alpha(1) #deals with transparency



        #axes[0].set_title("ROAR CA WebApp Output: Blanks", weight='bold')
        axes[0].set_title(titleText + ": Blanks", weight='bold')
        axes[0].set_xlabel("Mean Abundance", fontsize = 12)
        axes[0].set_ylabel("CV", fontsize = 12)
        axes[0].set_ylim(0,4)
        axes[0].set_xlim(100, 10000000000)
        axes[0].set(xscale='log')
        axes[0].set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5,3.0,3.5,4.0])

        b=sns.scatterplot(data=plot2.loc[((plot2['type']!='blank')),:],
                        x='Mean', y='CV', color="whitesmoke",
                        edgecolor = 'black', alpha = 0.5, ax = axes[1])

        c=sns.scatterplot(data=plot2.loc[((plot2['type']!='blank')&(plot2['spike']==1)),:],
                        x='Mean', y='CV', color="firebrick",
                        edgecolor = 'black', alpha = 0.5, ax = axes[1])

        # palette = ['royalblue', 'gold']
        # sns.set_palette(palette, 2)

        # d=sns.scatterplot(data=plot2.loc[((plot2['type']!='blank')&(plot2['present']==1)),:],
        #                 x='Mean', y='CV', hue = 'native',
        #                 edgecolor = 'black', alpha = 0.5, ax = axes[1])

        # c.axhline(y=1.25, color='red', linestyle="dashed", linewidth=1.5, alpha=1)
        # c.text(1000000000, 1.4, 'CV = 1.25', ha='center', va='center_baseline', weight='bold', size = 12)
        c.axhline(y=max_replicate_cv_value, color='red', linestyle="dashed", linewidth=1.5, alpha=1)
        c.text(1000000000, max_replicate_cv_value+0.1, 'CV = {}'.format(max_replicate_cv_value), ha='center', va='center_baseline', weight='bold', size = 12)

        #axes[1].set_title("ROAR CA WebApp Output: Samples", weight='bold')
        axes[1].set_title(titleText + ": Samples", weight='bold')
        axes[1].set_xlabel("Mean Abundance", fontsize = 12)
        axes[1].set_ylabel("CV", fontsize = 12)
        axes[1].set_ylim(0,4)
        axes[1].set_xlim(100, 10000000000)
        axes[1].set(xscale='log')
        axes[1].set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5,3.0,3.5,4.0])
        # legend = d.legend(title = "Natives")
        # legend.get_texts()[0].set_text('present')
        # legend.get_texts()[1].set_text('spiked')

        # frame = legend.get_frame() #sets up for color, edge, and transparency
        # frame.set_facecolor('lightgray') #color of legend
        # frame.set_edgecolor('black') #edge color of legend
        # frame.set_alpha(1) #deals with transparency

        # plt.savefig('./outputTest02/cv_scatterplot.png', bbox_inches='tight')

        # Convert the plot to a bytes-like object
        buffer = io.BytesIO()
        plt.savefig(buffer)
        # buffer.seek(0)

        self.cv_scatterplots_out.append(buffer.getvalue())

        self.cv_scatterplot_map['cv_scatterplot'] = self.cv_scatterplots_out[0]

        project_name = self.parameters['project_name'][1] 
        self.gridfs.put("&&".join(self.cv_scatterplot_map.keys()), _id=self.jobid + "_cv_scatterplots", encoding='utf-8', project_name = project_name)
        self.mongo_save(self.cv_scatterplot_map['cv_scatterplot'], step='cv_scatterplot')

        # reset plt
        plt.clf()


    def occurrence_heatmap(self, input_dfs):
        plt.rcdefaults()

        max_replicate_cv_value = self.parameters['max_replicate_cv'][1]
        min_replicate_hits_percent = self.parameters['min_replicate_hits'][1]
        titleText = "Heatmap of all feature occurrences using filter values of {} maximum CV and {} minimum replicate %".format(max_replicate_cv_value, min_replicate_hits_percent)
        
        # convert max_replicate_cv_value to a numeric value
        max_replicate_cv_value = pd.to_numeric(self.parameters['max_replicate_cv'][1], errors='coerce')

        # convert min_replicate_hits_percent to a numeric value
        min_replicate_hits_percent = pd.to_numeric(self.parameters['min_replicate_hits'][1], errors='coerce')

        # get dataframe 'Feature_statistics_positive' if it exists else None
        dfPos = self.data_map['Feature_statistics_positive'] if 'Feature_statistics_positive' in self.data_map else None

        # get dataframe 'Feature_statistics_negative' if it exists else None  
        dfNeg = self.data_map['Feature_statistics_negative'] if 'Feature_statistics_negative' in self.data_map else None

        # combine the two dataframes. Ignnore non-existing dataframes
        dfCombined = pd.concat([dfPos, dfNeg], axis=0, ignore_index=True, sort=False) if dfPos is not None and dfNeg is not None else dfPos if dfPos is not None else dfNeg if dfNeg is not None else None

        # # log the dataframes if not None
        # if dfPos is not None:
        #     logger.info("dfPos= {}".format(dfPos.columns.values))
        # if dfNeg is not None:
        #     logger.info("dfNeg= {}".format(dfNeg.columns.values))
        # if dfCombined is not None:
        #     logger.info("dfCombined= {}".format(dfCombined.columns.values))

        # Get sample headers                
        all_headers = task_fun.parse_headers(dfCombined) 
        #logger.info("all_headers= {}".format(all_headers))
        sam_headers = [sublist[0][:-1] for sublist in all_headers if len(sublist) > 1]
        #logger.info("sam_headers= {}".format(sam_headers))

        # Isolate sample_groups from stats columns
        prefixes = ['Mean_','Median_', 'CV_', 'STD_', 'N_Abun_', 'Replicate_Percent_']
        sample_groups = [item for item in sam_headers if not any(x in item for x in prefixes)]
        logger.info("sample_groups= {}".format(sample_groups))

        # Blank_MDL - need to check what the blank samples are actually named
        blank_strings = ['MB', 'Mb', 'mb', 'BLANK', 'Blank', 'blank', 'BLK', 'Blk']
        blank_col = [item for item in sample_groups if any(x in item for x in blank_strings)]
        logger.info("blank_col= {}".format(blank_col))

        blank_mean = 'Mean_' + blank_col[0]
        #logger.info("blank_mean= {}".format(blank_mean))
        blank_std = 'STD_' + blank_col[0]
        #logger.info("blank_std= {}".format(blank_std))

        dfCombined['MDL'] = dfCombined[blank_mean] + 3*dfCombined[blank_std]
        #logger.info("dfCombined['MDL']= {}".format(dfCombined['MDL']))
        #logger.info("dfCombined= {}".format(dfCombined.columns.values))

        # Find CV cols from df
        cv_cols = ['CV_' + col for col in sample_groups]
        #logger.info("cv_cols= {}".format(cv_cols))
        rper_cols = ['Replicate_Percent_' + col for col in sample_groups]
        #logger.info("rper_cols= {}".format(rper_cols))
        med_cols = ['Median_' + col for col in sample_groups]
        #logger.info("med_cols= {}".format(med_cols))

        # Grab CV cols from df
        cv_df = dfCombined[cv_cols]
        #logger.info("cv_df= {}".format(cv_df.columns.values))
        rper_df = dfCombined[rper_cols]
        #logger.info("rper_df= {}".format(rper_df.columns.values))
        med_df = dfCombined[med_cols]
        #logger.info("med_df= {}".format(med_df.columns.values))

        logger.info("cv_df shape= {}".format(cv_df.shape))
        logger.info("rper_df shape= {}".format(rper_df.shape))
        logger.info("med_df shape= {}".format(med_df.shape))
        logger.info("dfCombined shape= {}".format(dfCombined.shape))

        # Blank out cvs in samples with <2 samples -- NEED TO UPDATE TO REPLICATE PERCENT
        for x,y,z in zip(cv_cols, rper_cols, med_cols):
            # Replace cv_df values with nan in cv_col for n_abun and MDL cutoffs
            cv_df.loc[dfCombined[y]<min_replicate_hits_percent, x] = np.nan
            cv_df.loc[dfCombined[z]<=dfCombined['MDL'], x] = np.nan
        #logger.info("#1 cv_df= {}".format(cv_df.values))

        # Add sum of Trues for condition applied to cv dataframe
        cv_df['below count'] = (cv_df <= max_replicate_cv_value).sum(axis=1)
        #logger.info("#2 cv_df= {}".format(cv_df.values))

        # Sort values by medians
        cv_df = cv_df.sort_values('below count')
        #logger.info("#3 cv_df= {}".format(cv_df.values))

        # Remove median column
        del cv_df[cv_df.columns[-1]]
        #logger.info("#4 cv_df= {}".format(cv_df.values))

        # Create masks for CV cutoffs
        above = cv_df > max_replicate_cv_value
        below = cv_df <= max_replicate_cv_value
        nan_ = cv_df.isna()

        # Get sum values for each group (use these in colorbar labels)
        print(above.sum().sum())
        print(below.sum().sum())
        print(nan_.sum().sum())

        # Use masks to changes values in cv_df to 1, 0, -1
        dum = np.where(above, 1, cv_df)
        dum = np.where(below, 0, dum)
        dum = np.where(nan_, -1, dum)
        logger.info("#5 cv_df= {}".format(cv_df.values))

        # Create matrix from discretized dataframe
        cv_df_discrete = pd.DataFrame(dum, index=cv_df.index, columns=cv_df.columns)
        cv_df_trans = cv_df_discrete.transpose()

        # Set Figure size and title
        plt.figure(figsize = (40,15))
        plt.title(titleText, fontsize = 36)
        # plt.title('CA Data MS1 Feature Heatmap (post-blank subtraction), ESI- (n='+str(cv_df.size)+')\nSorted by # samples below CV threshold (lowest[left] to highest[right])', fontsize = 36)

        # Create custom color mapping
        myColors = ((0.8, 0.8, 0.8, 1.0), (1.0, 1.0, 1.0, 1.0), (1, 0.0, 0.2, 1.0))
        cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))

        # Plot heatmap
        ax = sns.heatmap(cv_df_trans, cmap=cmap, cbar_kws={"shrink": 0.2, "pad": 0.01})

        ax.set(xticklabels=[])

        # Manually specify colorbar labelling after it's been generated
        colorbar = ax.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=24)
        colorbar.set_ticks([-0.667, 0, 0.667])
        colorbar.set_ticklabels([
            'non detect ({})'.format(nan_.sum().sum()),
            'CV <= {} ({})'.format(max_replicate_cv_value, below.sum().sum()),
            'CV > {} ({})'.format(max_replicate_cv_value, above.sum().sum())])


        plt.ylabel('Sample Set', fontsize = 28)
        plt.xlabel('Feature ID (n = '+str(len(cv_df))+')', fontsize = 28)

        # plt.savefig('./outputTest02/occurrence_heatmap.png', bbox_inches='tight')

        # Convert the plot to a bytes-like object
        buffer = io.BytesIO()
        plt.savefig(buffer)
        # buffer.seek(0)

        self.occurrence_heatmaps_out.append(buffer.getvalue())

        self.occurrence_heatmap_map['occurrence_heatmap'] = self.occurrence_heatmaps_out[0]

        project_name = self.parameters['project_name'][1] 
        self.gridfs.put("&&".join(self.occurrence_heatmap_map.keys()), _id=self.jobid + "_occurrence_heatmaps", encoding='utf-8', project_name = project_name)
        self.mongo_save(self.occurrence_heatmap_map['occurrence_heatmap'], step='occurrence_heatmap')

        # reset plt
        plt.clf()


    def check_tracers(self):
        if self.tracer_df is None:
            logger.info("No tracer file, skipping this step.")
            return
        if self.verbose:
            logger.info("Tracer file found, checking tracers.")
        ppm = self.parameters['mass_accuracy_units_tr'][1]== 'ppm'
        mass_accuracy_tr = float(self.parameters['mass_accuracy_tr'][1])
        self.tracer_dfs_out = [task_fun.check_feature_tracers(df, self.tracer_df, mass_accuracy_tr, float(self.parameters['rt_accuracy_tr'][1]), ppm)[0] if df is not None else None for index, df in enumerate(self.dfs)]
        self.dfs = [task_fun.check_feature_tracers(df, self.tracer_df, mass_accuracy_tr, float(self.parameters['rt_accuracy_tr'][1]), ppm)[1] if df is not None else None for index, df in enumerate(self.dfs)]
        #logger.info("self.tracer_dfs_out[0].shape = {}".format(self.tracer_dfs_out[0].shape))
        
        self.tracer_dfs_out = [format_tracer_file(df) if df is not None else None for df in self.tracer_dfs_out]
        # self.tracer_plots_out = [create_tracer_plot(df) for df in self.tracer_dfs_out]
        
        #logger.info("self.tracer_dfs_out[1].shape = {}".format(self.tracer_dfs_out[1].shape))

        # declare plotter
        df_WA = WebApp_plotter()

        # logger.info("self.tracer_dfs_out[0].shape = {}".format(self.tracer_dfs_out[0].shape))
        # logger.info("self.tracer_dfs_out[0].columns.values = {}".format(self.tracer_dfs_out[0].columns.values))

        # plot
        if self.tracer_dfs_out[0] is not None:
            listOfPNGs,df_debug = df_WA.make_seq_scatter(
                # data_path='./input/summary_tracer.xlsx',
                df_in=self.tracer_dfs_out[0],   
                seq_csv=self.run_sequence_pos_df,                        
                ionization='pos',
                y_scale='log',
                fit=False,
                share_y=False,
                y_fixed=False,
                y_step=6,
                same_frame=False,
                legend=True,
                chemical_names=None, 
                # save_image=True, 
                # image_title='./output02/slide_12-dark',
                dark_mode=True)
            
            logger.info("df_debug= {}".format(df_debug.columns.values))

            self.tracer_plots_out.append(listOfPNGs)
    
        else:
            self.tracer_plots_out.append(None)
    
    
        # declare plotter
        df_WA = WebApp_plotter()
        # df_WA,df_debug = WebApp_plotter()
        # logger.info("df_debug= {}".format(df_debug.columns.values))

        # plot
        if self.tracer_dfs_out[1] is not None:
            listOfPNGs, df_debug = df_WA.make_seq_scatter(
                # data_path='./input/summary_tracer.xlsx',
                df_in=self.tracer_dfs_out[1],                           
                seq_csv=self.run_sequence_neg_df,
                ionization='neg',
                y_scale='log',
                fit=False,
                share_y=False,
                y_fixed=False,
                y_step=6,
                same_frame=False,
                legend=True,
                chemical_names=None, 
                # save_image=True, 
                # image_title='./output02/slide_12-dark',
                dark_mode=True)
            
            #logger.info("df_debug= {}".format(df_debug.columns.values))

            self.tracer_plots_out.append(listOfPNGs)
            logger.info("df_debug.head(5)= {}".format(df_debug.head(5)))
        else:
            self.tracer_plots_out.append(None)
        
        # implements part of NTAW-143
        dft = pd.concat([self.tracer_dfs_out[0], self.tracer_dfs_out[1]])

        # remove the columns 'Detection_Count(non-blank_samples)' and 'Detection_Count(non-blank_samples)(%)'
        #dft = dft.drop(columns=['Detection_Count(non-blank_samples)','Detection_Count(non-blank_samples)(%)'])
        
        self.data_map['Tracer_Sample_Results'] = task_fun.column_sort_TSR(dft)


        # create summary table
        # AC 12/7/2023: commented out to move to function after clean_features
        # if 'DTXSID' not in dft.columns:
        #     dft['DTXSID'] = ''
        # dft = dft[['Feature_ID', 'Chemical_Name', 'DTXSID', 'Ionization_Mode', 'Mass_Error_PPM', 'Retention_Time_Difference', 'Max_CV_across_sample']]
        # self.data_map['Tracers_Summary'] = dft
        
        # self.tracer_map['tracer_plot_pos'] = self.tracer_plots_out[0]
        # self.tracer_map['tracer_plot_neg'] = self.tracer_plots_out[1]

        if self.tracer_plots_out[0] is not None:
            for i in range (len(self.tracer_plots_out[0])):
                self.tracer_map['tracer_plot_pos_'+str(i+1)] = self.tracer_plots_out[0][i]
            
        #logger.info(len(self.tracer_plots_out[1]))
        
        # Add an if statement below to account for: if only negative mode data is entered, and only a negative tracer file is submitted, tracer_plots_out will only have one entry at [0]
        if len(self.tracer_plots_out) > 1:
            if self.tracer_plots_out[1] is not None:
                for i in range (len(self.tracer_plots_out[1])):
                    self.tracer_map['tracer_plot_neg_'+str(i+1)] = self.tracer_plots_out[1][i]
         
        project_name = self.parameters['project_name'][1] 
        self.gridfs.put("&&".join(self.tracer_map.keys()), _id=self.jobid + "_tracer_files", encoding='utf-8', project_name = project_name)
        for key in self.tracer_map.keys():
            self.mongo_save(self.tracer_map[key], step=key)
        return


    def clean_features(self):
        controls = [float(self.parameters['min_replicate_hits'][1]), float(self.parameters['max_replicate_cv'][1]), float(self.parameters['min_replicate_hits_blanks'][1])]
        tracer_df_bool=False
        if self.tracer_df is not None:
            tracer_df_bool=True
        self.docs = [task_fun.clean_features(df, controls, tracer_df=tracer_df_bool)[1] if df is not None else None for index, df in enumerate(self.dfs)]
        self.dfs_flagged = [task_fun.clean_features(df, controls, tracer_df=tracer_df_bool)[2] if df is not None else None for index, df in enumerate(self.dfs)]
        self.dfs = [task_fun.clean_features(df, controls, tracer_df=tracer_df_bool)[0] if df is not None else None for index, df in enumerate(self.dfs)]
        #self.dfs, self.docs = map(list, zip(*[task_fun.clean_features(df, controls) if df is not None else None for index, df in enumerate(self.dfs)]))
        self.dfs = [task_fun.Blank_Subtract_Mean(df) if df is not None else None for index, df in enumerate(self.dfs)]  # subtract blanks from medians
        self.dfs_flagged = [task_fun.Blank_Subtract_Mean(df) if df is not None else None for index, df in enumerate(self.dfs_flagged)]  # subtract blanks from medians
        #self.mongo_save(self.dfs[0], FILENAMES['cleaned'][0])
        #self.mongo_save(self.dfs[1], FILENAMES['cleaned'][1])
        return
    
    
    def merge_columns_onto_tracers(self):
        # self.data_map['Tracer_Sample_Results'] = task_fun.column_sort_TSR(dft)
        if self.tracer_df is None:
            logger.info("No tracer file, skipping this step.")
            return
        # Grab the tracer dataframe from self.data_map
        dft = self.data_map['Tracer_Sample_Results']
        logger.info("dft: {}".format(dft.columns))
        '''
        # Go through both negative and positive mode dataframes combine when present into a new temp dataframe
        if self.dfs[0] is not None and self.dfs[1] is not None:
            temp_df = pd.concat([self.dfs[0], self.dfs[1]], axis=0)
        elif self.dfs[0] is not None:
            temp_df = self.dfs[0]
        else:
            temp_df = self.dfs[1]
              
        # Merge combined modes dataframe with tracer dataframe            
        dft = pd.merge(dft, temp_df[['Feature_ID', 'Occurrence_Count(all_samples)', 'Occurrence_Count(all_samples)(%)']], how='left', on='Feature_ID')
        
        # Push new version of tracers dataframe back into data_map
        self.data_map['Tracer_Sample_Results'] = dft

        logger.info("dft post-merge: {}".format(dft.columns))
        '''
        # create summary table
        if 'DTXSID' not in dft.columns:
            dft['DTXSID'] = ''
        dft = dft[['Feature_ID', 'Chemical_Name', 'DTXSID', 'Ionization_Mode', 'Mass_Error_PPM', 'Retention_Time_Difference', 'Max_CV_across_sample', 
        'Occurrence_Count(across_all_replicates)', 'Occurrence_Count(across_all_replicates)(%)']]
        self.data_map['Tracers_Summary'] = dft
        return
    

    def create_flags(self):
        self.dfs = [task_fun.flags(df) if df is not None else None for df in self.dfs]
        #self.mongo_save(self.dfs[0], FILENAMES['flags'][0])
        #self.mongo_save(self.dfs[1], FILENAMES['flags'][1])


    def combine_modes(self):
        tracer_df_bool=False
        if self.tracer_df is not None:
            tracer_df_bool=True
        self.df_combined = task_fun.combine(self.dfs[0], self.dfs[1])
        self.df_flagged_combined = task_fun.combine(self.dfs_flagged[0], self.dfs_flagged[1])
        self.doc_combined = [task_fun.combine_doc(doc, dupe, tracer_df=tracer_df_bool) if doc is not None else None for doc, dupe in zip(self.docs, self.dupes)]
        if self.doc_combined[0] is not None and self.doc_combined[1] is not None:
            self.doc_combined = pd.concat([self.doc_combined[0], self.doc_combined[1]], axis=0)
        elif self.doc_combined[0] is not None:
            self.doc_combined = self.doc_combined[0]
        else:
            self.doc_combined = self.doc_combined[1]
        self.data_map['Filter_documentation'] = self.doc_combined
        #self.mongo_save(self.df_combined, FILENAMES['combined'])
        self.mpp_ready = task_fun.MPP_Ready(self.df_combined, self.pass_through, tracer_df_bool)
        self.mpp_ready_flagged = task_fun.MPP_Ready(self.df_flagged_combined, self.pass_through, tracer_df_bool)
        #self.data_map['Cleaned_feature_results_full'] = remove_columns(self.mpp_ready,['Occurrence_Count(all_samples)','Occurrence_Count(all_samples)(%)'])
        self.data_map['Cleaned_feature_results_reduced'] = reduced_file(self.mpp_ready)
        self.data_map['Results_flagged'] = reduced_file(self.mpp_ready_flagged)


    def perform_dashboard_search(self, lower_index=0, upper_index=None, save = True):
        logging.info('Rows flagged for dashboard search: {} out of {}'.format(len(self.df_combined.loc[self.df_combined['For_Dashboard_Search'] == '1', :]), len(self.df_combined)))
        to_search = self.df_combined.loc[self.df_combined['For_Dashboard_Search'] == '1', :].copy()  # only rows flagged
        if self.parameters['search_mode'][1] == 'mass':
            to_search.drop_duplicates(subset='Mass', keep='first', inplace=True)
        else:
            # NTAW-94 rename compound to formula
            #to_search.drop_duplicates(subset='Compound', keep='first', inplace=True)
            to_search.drop_duplicates(subset='Formula', keep='first', inplace=True)
        n_search = len(to_search)  # number of fragments to search
        logger.info("Total # of queries: {}".format(n_search))
        #max_search = 300  # the maximum number of fragments to search at a time
        #upper_index = 0
        to_search = to_search.iloc[lower_index:upper_index, :]
        if self.parameters['search_mode'][1] == 'mass':
            mono_masses = task_fun.masses(to_search)
            dsstox_search_df = api_search_masses_batch(mono_masses, float(self.parameters['parent_ion_mass_accuracy'][1]),
                                                       batchsize=int(self.parameters['api_batch_size'][1]), jobid=self.jobid)
        else:
            formulas = task_fun.formulas(to_search)
            response = api_search_formulas(formulas, self.jobid)
            if not response.ok: # check if we got a successful response
                raise requests.exceptions.HTTPError("Unable to access DSSTOX API. Please contact an administrator or try turning the DSSTox search option off.")
            dsstox_search_json = io.StringIO(json.dumps(response.json()['results']))
            dsstox_search_df = pd.read_json(dsstox_search_json, orient='split',
                                            dtype={'TOXCAST_NUMBER_OF_ASSAYS/TOTAL': 'object'})
        dsstox_search_df = self.mpp_ready[['Feature_ID','Mass', 'Retention_Time']].merge(dsstox_search_df, how = 'right', left_on = 'Mass', right_on = 'INPUT')
        self.data_map['chemical_results'] = dsstox_search_df
        self.search_results = dsstox_search_df

        
    def perform_hcd_search(self):
        logger.info(f'Querying the HCD with DTXSID identifiers')
        if len(self.search_results) > 0:
            dtxsid_list = self.search_results['DTXSID'].unique()
            hcd_results = batch_search_hcd(dtxsid_list)
            self.search_results = self.search_results.merge(hcd_results, how = 'left', on = 'DTXSID')
            self.data_map['chemical_results'] = self.search_results
            self.data_map['hcd_search'] = hcd_results
    
    
    def store_data(self):
        logger.info(f'Storing data files to MongoDB')
        project_name = self.parameters['project_name'][1] 
        self.gridfs.put("&&".join(self.data_map.keys()), _id=self.jobid + "_file_names", encoding='utf-8', project_name = project_name)
        for key in self.data_map.keys():
            self.mongo_save(self.data_map[key], step=key)


    def process_toxpi(self):
        by_mass = self.parameters['search_mode'][1] == "mass"
        self.df_combined = toxpi.process_toxpi(self.df_combined, self.search_results,
                                               tophit=(self.parameters['top_result_only'][1] == 'yes'), by_mass = by_mass)
        self.data_map['final_full'] = self.df_combined
        self.data_map['final_reduced'] = reduced_file(self.df_combined)


    def mongo_save(self, file, step=""):
        
        if isinstance(file, pd.DataFrame):
            to_save = file.to_json(orient='split')
        else:
            to_save = file
        id = self.jobid + "_" + step
        
        project_name = self.parameters['project_name'][1] 
        self.gridfs.put(to_save, _id=id, encoding='utf-8', project_name = project_name)
