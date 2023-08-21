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
        self.df_combined = None
        self.mpp_ready = None
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
        #self.data_dir = os.path.join(self.base_dir, 'data', self.jobid)
        #self.new_download_dir = os.path.join(self.data_dir, "new")
        self.step = "Started"  # tracks the current step (for fail messages)
        #os.mkdir(self.data_dir)
        #os.mkdir(self.new_download_dir)
        self.tracer_plots_out = []


    def execute(self):

        # 0: create a status in mongo
        self.set_status('Processing', create = True)

        #0: create an analysis_parameters sheet
        self.create_analysis_parameters_sheet()

        # 1: drop duplicates and throw out void volume
        self.step = "Dropping duplicates"
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
        self.dfs = [task_fun.cal_detection_count(df) if df is not None else None for df in self.dfs]

        logger.info("after task_fun.cal_detection_count self.dfs.size(): {}".format(len(self.dfs)))
 
        # 3: check tracers (optional)
        self.step = "Checking tracers"
        self.check_tracers()
        if self.verbose:
            logger.info("Checked tracers.")
            #print(self.tracer_dfs_out)
        # counting occrrences of each feature after cleaning

        if self.dfs[1] is not None:
            logger.info("Self.df columns: {}".format(self.dfs[1].columns.values))
            logger.info("before clean_features self.dfs.size(): {}".format(len(self.dfs)))

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

        # 5: create flags
        self.step = "Creating flags"
        self.create_flags()
        if self.verbose:
            logger.info("Created flags.")
            if self.dfs[0] is not None:
                logger.info("POS df length: {}".format(len(self.dfs[0])))
            if self.dfs[1] is not None:
                logger.info("NEG df length: {}".format(len(self.dfs[1])))
            #print(self.dfs[0])

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
            self.process_toxpi()
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

    def filter_duplicates(self):
        # self.dfs = [task_fun.duplicates(df) for df in self.dfs if df is not None, None]
        self.dfs = [task_fun.duplicates(df) if df is not None else None for df in self.dfs]
        return

    def filter_void_volume(self, min_rt):
        # self.dfs = [df.loc[df['Retention_Time'] > min_rt].copy() for index, df in enumerate(self.dfs) if df is not None, None]
        self.dfs = [df.loc[df['Retention_Time'] > min_rt].copy() if df is not None else None for index, df in enumerate(self.dfs)]
        return

    def calc_statistics(self):
        ppm = self.parameters['mass_accuracy_units'][1]== 'ppm'
        self.dfs = [task_fun.statistics(df) if df is not None else None for df in self.dfs]
        if self.dfs[0] is not None and self.dfs[1] is not None:
            self.dfs[0] = task_fun.assign_feature_id(self.dfs[0])
            self.dfs[1] = task_fun.assign_feature_id(self.dfs[1], start=len(self.dfs[0].index)+1)
            mass_accuracy = float(self.parameters['mass_accuracy'][1])
            rt_accuracy = float(self.parameters['rt_accuracy'][1])
            self.dfs[0] = task_fun.adduct_identifier(self.dfs[0], mass_accuracy, rt_accuracy, ppm,
                                                 ionization='positive', id_start=1)
            self.dfs[1] = task_fun.adduct_identifier(self.dfs[1], mass_accuracy, rt_accuracy, ppm,
                                                 ionization='negative', id_start=len(self.dfs[0].index)+1)
            self.data_map['Feature_statistics_positive'] = self.dfs[0]
            self.data_map['Feature_statistics_negative'] = self.dfs[1]
        elif self.dfs[0] is not None:
            mass_accuracy = float(self.parameters['mass_accuracy'][1])
            self.dfs[0] = task_fun.assign_feature_id(self.dfs[0])
            rt_accuracy = float(self.parameters['rt_accuracy'][1])
            self.dfs[0] = task_fun.adduct_identifier(self.dfs[0], mass_accuracy, rt_accuracy, ppm,
                                                 ionization='positive', id_start=1)
            self.data_map['Feature_statistics_positive'] = self.dfs[0]
        else:
            mass_accuracy = float(self.parameters['mass_accuracy'][1])
            self.dfs[1] = task_fun.assign_feature_id(self.dfs[1])
            rt_accuracy = float(self.parameters['rt_accuracy'][1])
            self.dfs[1] = task_fun.adduct_identifier(self.dfs[1], mass_accuracy, rt_accuracy, ppm,
                                                 ionization='negative', id_start=1)
            self.data_map['Feature_statistics_negative'] = self.dfs[1]
        return

    def check_tracers(self):
        if self.tracer_df is None:
            logger.info("No tracer file, skipping this step.")
            return
        if self.verbose:
            logger.info("Tracer file found, checking tracers.")
        ppm = self.parameters['mass_accuracy_units_tr'][1]== 'ppm'
        mass_accuracy_tr = float(self.parameters['mass_accuracy_tr'][1])
        self.tracer_dfs_out = [fn.check_feature_tracers(df, self.tracer_df, mass_accuracy_tr, float(self.parameters['rt_accuracy_tr'][1]), ppm) if df is not None else None for index, df in enumerate(self.dfs)]
        self.tracer_dfs_out = [format_tracer_file(df) if df is not None else None for df in self.tracer_dfs_out]
        # self.tracer_plots_out = [create_tracer_plot(df) for df in self.tracer_dfs_out]

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
                y_scale='linear',
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
                y_scale='linear',
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
    
        # implements part of NTAW-143
        dft = pd.concat([self.tracer_dfs_out[0], self.tracer_dfs_out[1]])

        # remove the columns 'Detection_Count(non-blank_samples)' and 'Detection_Count(non-blank_samples)(%)'
        dft = dft.drop(columns=['Detection_Count(non-blank_samples)','Detection_Count(non-blank_samples)(%)'])
        
        self.data_map['Tracer_Sample_Results'] = dft


        # create summary table
        if 'DTXSID' not in dft.columns:
            dft['DTXSID'] = ''
        dft = dft[['Chemical_Name', 'DTXSID', 'Ionization_Mode', 'Mass_Error_PPM', 'Retention_Time_Difference', 'Max_CV_across_sample', 'Detection_Count(all_samples)','Detection_Count(all_samples)(%)']]
        self.data_map['Tracers_Summary'] = dft
        
        # self.tracer_map['tracer_plot_pos'] = self.tracer_plots_out[0]
        # self.tracer_map['tracer_plot_neg'] = self.tracer_plots_out[1]

        for i in range (len(self.tracer_plots_out[0])):
            self.tracer_map['tracer_plot_pos_'+str(i+1)] = self.tracer_plots_out[0][i]
            
        logger.info(len(self.tracer_plots_out[1]))
        
        for i in range (len(self.tracer_plots_out[1])):
            self.tracer_map['tracer_plot_neg_'+str(i+1)] = self.tracer_plots_out[1][i]
         
        project_name = self.parameters['project_name'][1] 
        self.gridfs.put("&&".join(self.tracer_map.keys()), _id=self.jobid + "_tracer_files", encoding='utf-8', project_name = project_name)
        for key in self.tracer_map.keys():
            self.mongo_save(self.tracer_map[key], step=key)
        return

    def clean_features(self):
        controls = [float(self.parameters['min_replicate_hits'][1]), float(self.parameters['max_replicate_cv'][1]), float(self.parameters['min_replicate_hits_blanks'][1])]
        self.dfs = [task_fun.clean_features(df, controls) if df is not None else None for index, df in enumerate(self.dfs)]
        self.dfs = [fn.Blank_Subtract(df, index) if df is not None else None for index, df in enumerate(self.dfs)]  # subtract blanks from medians
        #self.mongo_save(self.dfs[0], FILENAMES['cleaned'][0])
        #self.mongo_save(self.dfs[1], FILENAMES['cleaned'][1])
        return

    def create_flags(self):
        self.dfs = [fn.flags(df) if df is not None else None for df in self.dfs]
        #self.mongo_save(self.dfs[0], FILENAMES['flags'][0])
        #self.mongo_save(self.dfs[1], FILENAMES['flags'][1])

    def combine_modes(self):

        self.df_combined = fn.combine(self.dfs[0], self.dfs[1])
        #self.mongo_save(self.df_combined, FILENAMES['combined'])
        self.mpp_ready = fn.MPP_Ready(self.df_combined)
        self.data_map['Cleaned_feature_results_full'] = remove_columns(self.mpp_ready,['Detection_Count(all_samples)','Detection_Count(all_samples)(%)'])
        self.data_map['Cleaned_feature_results_reduced'] = reduced_file(self.mpp_ready)

    def perform_dashboard_search(self, lower_index=0, upper_index=None, save = True):
        logging.info('Rows flagged for dashboard search: {} out of {}'.format(len(self.df_combined.loc[self.df_combined['For_Dashboard_Search'] == '1', :]), len(self.df_combined)))
        to_search = self.df_combined.loc[self.df_combined['For_Dashboard_Search'] == '1', :].copy()  # only rows flagged
        if self.parameters['search_mode'][1] == 'mass':
            to_search.drop_duplicates(subset='Mass', keep='first', inplace=True)
        else:
            to_search.drop_duplicates(subset='Compound', keep='first', inplace=True)
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
