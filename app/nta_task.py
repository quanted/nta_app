import pandas as pd
import json
import os
import csv
import time
import logging
from datetime import datetime
from dask.distributed import Client, LocalCluster, fire_and_forget

from . import functions_Universal_v3 as fn
from. import Toxpi_v3 as toxpi
from .batch_search_v3 import BatchSearch
from .utilities import connect_to_mongoDB

IN_DOCKER = os.environ.get("IN_DOCKER")
#IN_DOCKER = "False"  #for local

logger = logging.getLogger(__name__)


def run_nta_dask(parameters, input_dfs, tracer_df = None, jobid = "00000000", verbose = True):
    in_docker = IN_DOCKER != "False"
    if not in_docker:
        logger.info("Running in local development mode.")
        local_cluster = LocalCluster(processes=False)
        dask_client = Client(local_cluster)
    else:
        logger.info("Running in docker environment.")
        dask_client = Client('dask_scheduler:8786', processes=False)
    dask_input_dfs = dask_client.scatter(input_dfs)
    logger.info("Submitting Nta Dask task")
    task = dask_client.submit(run_nta, parameters, dask_input_dfs, tracer_df, jobid, verbose, in_docker = in_docker)
    fire_and_forget(task)
    #logger.info("Awaiting task completion")


def run_nta(parameters, input_dfs, tracer_df = None, jobid = "00000000", verbose = True, in_docker = True):
    nta_run = NtaRun(parameters, input_dfs, tracer_df, jobid, verbose, in_docker = in_docker)
    nta_run.execute()
    return True


FILENAMES = {'stats': ['stats_pos', 'stats_neg'],
             'tracers': ['tracers_pos', 'tracers_neg'],
             'cleaned': ['cleaned_pos', 'cleaned_neg'],
             'flags': ['flags_pos', 'flags_neg'],
             'combined': 'combined',
             'mpp_ready': 'combined_mpp_ready',
             'dashboard': 'dashboard_search',
             'toxpi': 'combined_toxpi'}

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
        self.entact = parameters['entact'] == "yes"
        self.sample_to_blank = float(parameters['sample_to_blank'])
        self.min_replicate_hits = float(parameters['min_replicate_hits'])
        self.max_replicate_cv = float(parameters['max_replicate_cv'])
        self.parent_ion_mass_accuracy = float(parameters['parent_ion_mass_accuracy'])
        self.search_mode = parameters['search_mode']
        self.top_result_only = parameters['top_result_only'] == 'yes'
        self.dfs = input_dfs
        self.df_combined = None
        self.mpp_ready = None
        self.search_results = None
        self.download_filename = None
        self.jobid = jobid
        self.verbose = verbose
        self.in_docker = in_docker
        self.mongo = connect_to_mongoDB(in_docker = self.in_docker)
        self.base_dir = os.path.abspath(os.path.join(os.path.abspath(__file__),"../.."))
        self.data_dir = os.path.join(self.base_dir, 'data', self.jobid)
        os.mkdir(self.data_dir)


    def execute(self):

        # 0: create a status in mongo
        self.set_status('Processing')

        # 1: drop duplicates
        self.drop_duplicates()
        if self.verbose:
            logger.info("Dropped duplicates.")
            #print(self.dfs[0])

        # 2: statistics
        self.calc_statistics()
        if self.verbose:
            logger.info("Calculated statistics.")
            #print(self.dfs[0])
            #print(str(list(self.dfs[0])))

        # 3: check tracers (optional)
        self.check_tracers()
        if self.verbose:
            logger.info("Checked tracers.")
            #print(self.tracer_dfs_out)

        # 4: clean features
        self.clean_features()
        if self.verbose:
            logger.info("Cleaned features.")
            #print(self.dfs[0])

        # 5: create flags
        self.create_flags()
        if self.verbose:
            logger.info("Created flags.")
            #print(self.dfs[0])

        # 6: combine modes
        self.combine_modes()
        if self.verbose:
            logger.info("Combined modes.")
            #print(self.df_combined)

        # 7: search dashboard
        self.search_dashboard()
        if self.verbose:
            logger.info("Searching Dashboard.")
        self.download_finished()
        if self.verbose:
            logger.info("Download finished.")
        self.fix_overflows()
        self.process_toxpi()
        if self.verbose:
            logger.info("Final result processed.")
        self.clean_files()
        if self.verbose:
            logger.info("Download files removed, processing complete.")

        # 8: set status to completed
        self.set_status('Completed')



    def set_status(self, status):
        posts = self.mongo.posts
        time_stamp = datetime.utcnow()
        id = self.jobid + "_" + "status"
        data = {'_id': id, 'date': time_stamp, 'status': status}
        posts.update_one({'_id': id},{'$set': {'_id': id,
                                              'date': time_stamp,
                                              'status': status}},
                         upsert=True)
        posts.replace_one(data, data, upsert=True)


    def drop_duplicates(self):
        self.dfs = [fn.duplicates(df, index) for index, df in enumerate(self.dfs)]
        #self.mongo_save(self.dfs[0], 'input_no_duplicates_pos')
        #self.mongo_save( self.dfs[1], 'input_no_duplicates_neg')
        return

    def calc_statistics(self):
        ppm = self.mass_accuracy_units == 'ppm'
        self.dfs = [fn.statistics(df, index) for index, df in enumerate(self.dfs)]
        #print("Calculating statistics with units: " + self.mass_accuracy_units)
        self.dfs = [fn.adduct_identifier(df, index, self.mass_accuracy, self.rt_accuracy, ppm) for index, df in enumerate(self.dfs)]
        #self.save_df_to_mongo('stats_pos', self.dfs[0])
        #self.save_df_to_mongo('stats_neg', self.dfs[1])
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
        self.mongo_save(self.mpp_ready, FILENAMES['mpp_ready'])

    def search_dashboard(self):
        if self.search_mode == 'mass':
            mono_masses = fn.masses(self.df_combined)
            mono_masses_str = [str(i) for i in mono_masses]
            self.search = BatchSearch(linux = self.in_docker)
            self.search.batch_search(masses=mono_masses_str, formulas=None, directory=self.data_dir, by_formula=False, ppm=self.parent_ion_mass_accuracy)
        else:
            compounds = fn.formulas(self.df_combined)
            self.search = BatchSearch(linux = self.in_docker)
            self.search.batch_search(masses=None, formulas=compounds, directory=self.data_dir)

    def download_finished(self):
        finished = False
        tries = 0
        while not finished and tries < 100:
            for filename in os.listdir(self.data_dir):
                if filename.startswith('ChemistryDashboard-Batch-Search') and not filename.endswith("part"):
                        self.download_filename = filename
                        finished = True
            time.sleep(1)
        if not finished:
            self.search.close_driver()
            raise Exception("Download from the CompTox Chemistry Dashboard failed!")
        print("This is what was downloaded: " + self.download_filename)
        self.search.close_driver()
        results_path = os.path.join(self.data_dir,self.download_filename)
        time.sleep(5) #waiting a second to make sure data is copied from the partial dl file
        self.search_results = pd.read_csv(results_path, sep='\t')
        self.mongo_save(self.search_results, FILENAMES['dashboard'])
        return self.download_filename

    def fix_overflows(self):
        """
        This function fixes an error seen in some comptox dashboard results, where a newline character is inserted into
        the middle of the chemical names, messing up the tsv file.
        :return:
        """
        if self.download_filename is not None:
            results_path = os.path.join(self.data_dir, self.download_filename)
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
        self.df_combined = toxpi.process_toxpi(self.df_combined, self.data_dir, self.download_filename,
                                               tophit=self.top_result_only, by_mass = by_mass)
        self.mongo_save(self.search_results, FILENAMES['toxpi'])

    def clean_files(self):
        path_to_remove = os.path.join(self.data_dir, self.download_filename)
        os.remove(path_to_remove)
        os.rmdir(self.data_dir)
        if self.verbose:
            logger.info("Cleaned up download file.")



    def mongo_save(self, file, step=""):
        to_save = json.loads(file.to_json(orient='split'))
        posts = self.mongo.posts
        time_stamp = datetime.utcnow()
        id = self.jobid + "_" + step
        data = {'_id': id, 'date': time_stamp, 'project_name': self.project_name,'data': to_save}
        posts.insert_one(data)



