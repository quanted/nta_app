import pandas as pd
import os
import csv
import time
import logging
import traceback
import shutil
from datetime import datetime
from dask.distributed import Client, LocalCluster, fire_and_forget

from . import functions_Universal_v3 as fn
from. import Toxpi_v3 as toxpi
from .batch_search_v3 import BatchSearch
from .utilities import connect_to_mongoDB, connect_to_mongo_gridfs, reduced_file
from . import task_functions as task_fun

#os.environ['IN_DOCKER'] = "False" #for local dev - also see similar switch in tools/output_access.py
NO_DASK = False  # set this to true to run locally without test (for debug purposes)

logger = logging.getLogger("nta_app")
logger.setLevel(logging.INFO)

def run_nta_dask(parameters, input_dfs, tracer_df = None, jobid = "00000000", verbose = True):
    in_docker = os.environ.get("IN_DOCKER") != "False"
    if NO_DASK:
        run_nta(parameters, input_dfs, tracer_df, jobid, verbose, in_docker = in_docker)
        return
    if not in_docker:
        logger.info("Running in local development mode.")
        logger.info("Detected OS is {}".format(os.environ.get("SYSTEM_NAME")))
        local_cluster = LocalCluster(processes=False, ip='127.0.0.1')
        dask_client = Client(local_cluster)
    else:
        dask_scheduler = os.environ.get("DASK_SCHEDULER")
        logger.info("Running in docker environment. Dask Scheduler: {}".format(dask_scheduler))
        dask_client = Client(dask_scheduler, processes=False)
    dask_input_dfs = dask_client.scatter(input_dfs)
    logger.info("Submitting Nta Dask task")
    task = dask_client.submit(run_nta, parameters, dask_input_dfs, tracer_df, jobid, verbose, in_docker = in_docker)
    fire_and_forget(task)


def run_nta(parameters, input_dfs, tracer_df = None, jobid = "00000000", verbose = True, in_docker = True):
    nta_run = NtaRun(parameters, input_dfs, tracer_df, jobid, verbose, in_docker = in_docker)
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


FILENAMES = {'duplicates': ['duplicates_dropped_pos', 'duplicates_dropped_neg'],
             'stats': ['stats_pos', 'stats_neg'],
             'tracers': ['tracers_pos', 'tracers_neg'],
             'cleaned': ['cleaned_pos', 'cleaned_neg'],
             'flags': ['flags_pos', 'flags_neg'],
             'combined': 'combined',
             'mpp_ready': ['for_stats_full', 'for_stats_reduced'],
             'dashboard': 'dashboard_search',
             'toxpi': ['final_output_full', 'final_output_reduced']
             }

class NtaRun:
    
    def __init__(self, parameters=None, input_dfs=None, tracer_df=None, jobid = "00000000", verbose = True, in_docker = True):
        logger.info("Initializing NtaRun Task")
        self.project_name = parameters['project_name']
        self.mass_accuracy = float(parameters['mass_accuracy'])
        self.mass_accuracy_units = parameters['mass_accuracy_units']
        self.rt_accuracy = float(parameters['rt_accuracy'])
        self.tracer_df = tracer_df
        self.tracer_dfs_out = None
        self.mass_accuracy_tr = float(parameters['mass_accuracy_tr'])
        self.mass_accuracy_units_tr = parameters['mass_accuracy_units_tr']
        self.rt_accuracy_tr = float(parameters['rt_accuracy_tr'])
        self.entact = False #parameters['entact'] == "yes"
        self.sample_to_blank = float(parameters['sample_to_blank'])
        self.min_replicate_hits = float(parameters['min_replicate_hits'])
        self.max_replicate_cv = float(parameters['max_replicate_cv'])
        self.parent_ion_mass_accuracy = float(parameters['parent_ion_mass_accuracy'])
        self.search_mode = parameters['search_mode']
        self.top_result_only = parameters['top_result_only'] == 'yes'
        self.minimum_rt = float(parameters['minimum_rt']) # throw out features below this (void volume)
        self.dfs = input_dfs
        self.df_combined = None
        self.mpp_ready = None
        self.search_results = None
        self.search = None
        self.download_filenames = []
        self.jobid = jobid
        self.verbose = verbose
        self.in_docker = in_docker
        self.mongo = connect_to_mongoDB(in_docker = self.in_docker)
        self.gridfs = connect_to_mongo_gridfs(in_docker = self.in_docker)
        self.base_dir = os.path.abspath(os.path.join(os.path.abspath(__file__),"../.."))
        self.data_dir = os.path.join(self.base_dir, 'data', self.jobid)
        self.new_download_dir = os.path.join(self.data_dir, "new")
        self.step = "Started"  # tracks the current step (for fail messages)
        os.mkdir(self.data_dir)
        os.mkdir(self.new_download_dir)


    def execute(self):

        # 0: create a status in mongo
        self.set_status('Processing', create = True)

        # 1: drop duplicates and throw out void volume
        self.step = "Dropping duplicates"
        self.filter_void_volume(self.minimum_rt)
        self.filter_duplicates()
        if self.verbose:
            logger.info("Dropped duplicates.")
            #print(self.dfs[0])

        # 2: statistics
        self.step = "Calculating statistics"
        self.calc_statistics()
        if self.verbose:
            logger.info("Calculated statistics.")
            #print(self.dfs[0])
            #print(str(list(self.dfs[0])))

        # 3: check tracers (optional)
        self.step = "Checking tracers"
        self.check_tracers()
        if self.verbose:
            logger.info("Checked tracers.")
            #print(self.tracer_dfs_out)

        # 4: clean features
        self.step = "Cleaning features"
        self.clean_features()
        if self.verbose:
            logger.info("Cleaned features.")
            #print(self.dfs[0])

        # 5: create flags
        self.step = "Creating flags"
        self.create_flags()
        if self.verbose:
            logger.info("Created flags.")
            #print(self.dfs[0])

        # 6: combine modes
        self.step = "Combining modes"
        self.combine_modes()
        if self.verbose:
            logger.info("Combined modes.")
            #print(self.df_combined)

        # 7: search dashboard
        self.step = "Searching dashboard"
        self.iterate_searches()
        self.process_toxpi()
        if self.verbose:
            logger.info("Final result processed.")
        self.clean_files()
        if self.verbose:
            logger.info("Download files removed, processing complete.")

        # 8: set status to completed
        self.step = "Displaying results"
        self.set_status('Completed')

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
        self.dfs = [fn.duplicates(df, index, high_res=True) for index, df in enumerate(self.dfs)]
        self.mongo_save(self.dfs[0], FILENAMES['duplicates'][0])
        self.mongo_save(self.dfs[1], FILENAMES['duplicates'][1])
        return

    def filter_void_volume(self, min_rt):
        self.dfs = [df.loc[df['Retention_Time'] > min_rt].copy() for index, df in enumerate(self.dfs)]
        return

    def calc_statistics(self):
        ppm = self.mass_accuracy_units == 'ppm'
        self.dfs = [fn.statistics(df, index) for index, df in enumerate(self.dfs)]
        self.dfs[0] = task_fun.assign_feature_id(self.dfs[0])
        self.dfs[1] = task_fun.assign_feature_id(self.dfs[1], start=len(self.dfs[0].index)+1)
        self.dfs[0] = task_fun.adduct_identifier(self.dfs[0], self.mass_accuracy, self.rt_accuracy, ppm,
                                                 ionization='positive', id_start=1)
        self.dfs[1] = task_fun.adduct_identifier(self.dfs[1], self.mass_accuracy, self.rt_accuracy, ppm,
                                                 ionization='negative', id_start=len(self.dfs[0].index))
        self.mongo_save(self.dfs[0], FILENAMES['stats'][0])
        self.mongo_save(self.dfs[1], FILENAMES['stats'][1])
        return


    def check_tracers(self):
        if self.tracer_df is None:
            logger.info("No tracer file, skipping this step.")
            return
        if self.verbose:
            logger.info("Tracer file found, checking tracers.")
        ppm = self.mass_accuracy_units_tr == 'ppm'
        self.tracer_dfs_out = [fn.check_feature_tracers(df, self.tracer_df, self.mass_accuracy_tr, self.rt_accuracy_tr, ppm) for index, df in enumerate(self.dfs)]
        self.mongo_save(self.tracer_dfs_out[0], FILENAMES['tracers'][0])
        self.mongo_save(self.tracer_dfs_out[1], FILENAMES['tracers'][1])

        return

    def clean_features(self):
        controls = [self.sample_to_blank, self.min_replicate_hits, self.max_replicate_cv]
        self.dfs = [fn.clean_features(df, index, self.entact, controls) for index, df in enumerate(self.dfs)]
        self.dfs = [fn.Blank_Subtract(df, index) for index, df in enumerate(self.dfs)]  # subtract blanks from medians
        self.mongo_save(self.dfs[0], FILENAMES['cleaned'][0])
        self.mongo_save(self.dfs[1], FILENAMES['cleaned'][1])
        return

    def create_flags(self):
        self.dfs = [fn.flags(df) for df in self.dfs]
        self.mongo_save(self.dfs[0], FILENAMES['flags'][0])
        self.mongo_save(self.dfs[1], FILENAMES['flags'][1])

    def combine_modes(self):
        self.df_combined = fn.combine(self.dfs[0], self.dfs[1])
        self.mongo_save(self.df_combined, FILENAMES['combined'])
        self.mpp_ready = fn.MPP_Ready(self.df_combined)
        self.mongo_save(self.mpp_ready, FILENAMES['mpp_ready'][0])
        self.mongo_save(reduced_file(self.mpp_ready), FILENAMES['mpp_ready'][1])  # save the reduced version


    def iterate_searches(self):
        to_search = self.df_combined.loc[self.df_combined['For_Dashboard_Search'] == '1', :].copy()  # only rows flagged
        if self.search_mode == 'mass':
            to_search.drop_duplicates(subset='Mass', keep='first', inplace=True)
        else:
            to_search.drop_duplicates(subset='Compound', keep='first', inplace=True)
        n_search = len(to_search)  # number of fragments to search
        logger.info("Total # of queries: {}".format(n_search))
        max_search = 300 # the maximum number of fragments to search at a time
        upper_index = 0
        finished = False
        while not finished:
            lower_index = upper_index
            upper_index = upper_index + max_search
            if upper_index >= (n_search - 1):
                upper_index = n_search - 1
                finished = True
            # finished = upper_index >= (n_search-1)
            if self.verbose:
                logger.info("Searching Dashboard for compounds {} - {}.".format(lower_index, upper_index))
            self.search_dashboard(to_search, lower_index, upper_index)
            self.download_finished(save=finished)
            if self.verbose:
                logger.info("Download finished.")
            self.fix_overflows()

    def search_dashboard(self, df_search, lower_index, upper_index):
        in_linux = os.environ.get("SYSTEM_NAME") != "WINDOWS"  # check for correct webdriver
        to_search = df_search.iloc[lower_index:upper_index, :]
        if self.search_mode == 'mass':
            mono_masses = fn.masses(to_search)
            mono_masses_str = [str(i) for i in mono_masses]
            self.search = BatchSearch(linux = in_linux)
            self.search.batch_search(masses=mono_masses_str, formulas=None, directory=self.new_download_dir,
                                                     by_formula=False, ppm=self.parent_ion_mass_accuracy)
        else:
            compounds = fn.formulas(to_search)
            self.search = BatchSearch(linux = in_linux)
            self.search.batch_search(masses=None, formulas=compounds, directory=self.new_download_dir)

    def download_finished(self, save = False):
        finished = False
        tries = 0
        while not finished and tries < 150:
            for filename in os.listdir(self.new_download_dir):
                if filename.startswith('ChemistryDashboard-Batch-Search') and not filename.endswith("part"):
                        self.download_filenames.append(filename)
                        copy_tries = 0
                        source = os.path.join(self.new_download_dir, filename)
                        destination = os.path.join(self.data_dir, filename)
                        while copy_tries < 10:
                            time.sleep(5)  # waiting a second to make sure data is copied from the partial dl file
                            shutil.copy(source, destination)
                            if os.path.exists(destination):
                                if os.path.getsize(destination) > 0:
                                    os.remove(source)
                                    break
                            copy_tries = copy_tries + 1
                        finished = True
            tries += 1
            time.sleep(1)
        if not finished:
            logger.info("Download never finished")
            self.search.close_driver()
            raise Exception("Download from the CompTox Chemistry Dashboard failed!")
        print("This is what was downloaded: " + self.download_filenames[-1])
        self.search.close_driver()
        results_path = os.path.join(self.data_dir,self.download_filenames[-1])
        self.fix_overflows(self.download_filenames[-1])
        download_df = pd.read_csv(results_path, sep='\t')
        if self.search_results is None:
            self.search_results = download_df
        else:
            self.search_results.append(download_df)
        if save:
            self.mongo_save(self.search_results, FILENAMES['dashboard'])
        return self.download_filenames[-1]

    def fix_overflows(self, filename=None):
        """
        This function fixes an error seen in some comptox dashboard results, where a newline character is inserted into
        the middle of the chemical names, messing up the tsv file.
        :return:
        """
        if  filename is not None:
            results_path = os.path.join(self.data_dir, filename)
            new_file = []
            problems = []
            row_len = []
            correct_len = 19
            with open(results_path, 'r') as file:
                reader = csv.reader(file, delimiter = '\t')
                for index, row in enumerate(reader):
                    row_len.append(len(row))
                    new_file.append(row)
                    last_line = row_len[index - 1]
                    if row_len[index] < correct_len and last_line < correct_len and row_len[index] + last_line == correct_len + 1:
                        new_file[index - 1][len(new_file[index - 1]) - 1] = new_file[index - 1][
                                                                                len(new_file[index - 1]) - 1] + \
                                                                            new_file[index][0]
                        new_file[index - 1] = new_file[index - 1] + new_file[index][1:]
                        problems.append(index)
            if problems: #check if list of problems is not empty
                [new_file.pop(i) for i in reversed(problems)]

            with open(results_path, 'w', newline = '') as file:
                writer = csv.writer(file, delimiter = '\t')
                writer.writerows(new_file)

    def process_toxpi(self):
        by_mass = self.search_mode == "mass"
        self.df_combined = toxpi.process_toxpi(self.df_combined, self.data_dir, self.download_filenames,
                                               tophit=self.top_result_only, by_mass = by_mass)
        self.mongo_save(self.df_combined, FILENAMES['toxpi'][0])
        self.mongo_save(reduced_file(self.df_combined), FILENAMES['toxpi'][1])


    def clean_files(self):
        shutil.rmtree(self.data_dir)  # remove data directory and all download files
        if self.verbose:
            logger.info("Cleaned up download file.")

    def mongo_save(self, file, step=""):
        to_save = file.to_json(orient='split')
        id = self.jobid + "_" + step
        self.gridfs.put(to_save, _id=id, encoding='utf-8', project_name = self.project_name)
