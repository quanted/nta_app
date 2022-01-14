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
from django.urls import reverse
from .utilities import connect_to_mongoDB, connect_to_mongo_gridfs, ms2_search_api
from .ms2_functions import sqlCFMID
from ..feature.feature import FeatureList
from ..feature.score_algo import SpectraScorer
from ...tools.ms2.send_email import send_ms2_finished


NO_DASK = False  # set this to True to run locally without dask (for debug purposes)

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("nta_app.ms2")
logger.setLevel(logging.INFO)


def run_ms2_dask(parameters, input_dfs, jobid = "00000000", verbose = True):
    in_docker = os.environ.get("IN_DOCKER") != "False"
    mongo_address = os.environ.get('MONGO_SERVER')
    link_address = reverse('ms2_results', kwargs={'jobid': jobid})
    if NO_DASK:
        run_ms2(parameters, input_dfs, mongo_address, jobid, results_link=link_address, verbose=verbose, in_docker=in_docker)
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
    dask_input_dfs = dask_client.scatter(input_dfs)
    logger.info("Submitting Nta ms2 Dask task")
    task = dask_client.submit(run_ms2, parameters, dask_input_dfs, mongo_address, jobid, results_link=link_address,
                              verbose=verbose, in_docker=in_docker)
    fire_and_forget(task)


def run_ms2(parameters, input_dfs,  mongo_address = None, jobid = "00000000", results_link = "", verbose = True,
            in_docker = True):
    ms2_run = MS2Run(parameters, input_dfs, mongo_address, jobid, results_link, verbose, in_docker = in_docker)
    try:
        ms2_run.execute()
    except Exception as e:
        trace = traceback.format_exc()
        logger.info(trace)
        fail_step = ms2_run.get_step()
        ms2_run.set_status("Failed on step: " + fail_step)
        error = repr(e)
        ms2_run.set_except_message(error)
        raise e
    return True


FILENAMES = {'final_output': ['CFMID_results_pos', 'CFMID_results_neg']}


class MS2Run:
    
    def __init__(self, parameters=None, input_dfs=None, mongo_address = None, jobid = "00000000",
                 results_link = None, verbose = True, in_docker = True):
        logger.info('MS2Run initialize - started')
        self.start = time.perf_counter()
        self.project_name = parameters['project_name']
        self.input_dfs = input_dfs
        self.n_masses = 1
        self.progress = 0
        self.results_dfs = [[None],[None]]
        self.features = {'pos' : [], 'neg' : []}
        #self.email = parameters['results_email']
        self.results_link = results_link
        self.precursor_mass_accuracy = float(parameters['precursor_mass_accuracy'])
        self.fragment_mass_accuracy = float(parameters['fragment_mass_accuracy'])
        self.jobid = jobid
        self.verbose = verbose
        self.in_docker = in_docker
        self.mongo_address = mongo_address
        self.mongo = connect_to_mongoDB(self.mongo_address)
        self.gridfs = connect_to_mongo_gridfs(self.mongo_address)
        self.step = "Started"  # tracks the current step (for fail messages)

    def execute(self):

        self.set_status('Reading Data', create= True)
        self.construct_featurelist()
                
        self.set_status('Retrieving Reference Spectra')
        self.get_CFMID_spectra();
        
        self.set_status('Preparing Results')
        self.mongo_save(self.features['pos'].to_df(), step=FILENAMES['final_output'][0])   
        self.mongo_save(self.features['neg'].to_df(), step=FILENAMES['final_output'][1])

        self.set_status('Completed')
        #self.send_email()
        logger.critical('Run Finished')
        logger.info(f'Run time: {time.perf_counter() - self.start}')

    def construct_featurelist(self):
        """
        Prepares pos- and neg-mode FeatureList object from the input dfs 
        """
        self.n_masses = len(self.input_dfs[0]) + len(self.input_dfs[1])
        for idx, df in enumerate(self.input_dfs):
            mode = 'pos' if idx == 0 else 'neg'
            tmp_feature_list = FeatureList()
            for data_block in df:
                tmp_feature_list.update_feature_list(data_block, POSMODE = mode == 'pos')
                self.update_progress()
            self.features[mode] = tmp_feature_list
    

    def get_CFMID_spectra(self):
        """
        Instantiate pos_list and neg_list with tuples of unique masses in the FeatureList and corrsponding mode. Iterate through list
        to get CFMID data. Returned spectra are appended to corresponding Features in the feature list using mass to join spectra.
        """
        pos_list = [(mass, 'ESI-MSMS-pos') for mass in self.features['pos'].get_masses(neutral = True)] if len(self.features['pos']) > 0 else [] 
        neg_list = [(mass, 'ESI-MSMS-neg') for mass in self.features['neg'].get_masses(neutral = True)] if len(self.features['neg']) > 0 else []
        self.n_masses = len(pos_list) + len(neg_list)
        for index, (mass, mode) in enumerate(pos_list + neg_list):
            logger.critical("Searching mass " + str(mass) + " number " + str(index) + " of " + str(len(pos_list + neg_list)))
            cfmid_result_list = ms2_search_api(mass, self.precursor_mass_accuracy, mode, self.jobid)
            self.update_progress()
            if cfmid_result_list is None:
                logger.critical(f'Found 0 structures for mass {mass}')
                continue
            logger.critical(f'Found {len(cfmid_result_list)} structures for mass {mass}')
            if len(cfmid_result_list) > 0:
                self.append_reference_spectra(mass, mode, cfmid_result_list)
        
    def append_reference_spectra(self, mass, mode, cfmid_result_list):
        """
        For each feature that corresponds to the input mass in the Feature List object, add a cfmid result
        and calculate the similarity scores using a SpectraScorer object.
        
        :param mass: mass used to find corresponding ms2 features in feature list
        :type mass: float
        :param mode: Value used to set the mass accuracy (ppm)
        :type mode: float, optional
        :param cfmid_result_list: List of cfmid_results returned from the api query
        :type cfmid_result_list: list of nested dictionaries
        """
        feature_mode = 'pos' if mode == 'ESI-MSMS-pos' else 'neg'
        spectra_scorer = SpectraScorer()
        for feature in self.features[feature_mode].get_features(mass, by='neutral_mass'):
            feature.add_reference_spectra(cfmid_result_list)
            feature.calc_similarity_scores(spectra_scorer)

    def send_email(self):
        try:
            #link_address = reverse('ms2_results', kwargs={'jobid': self.jobid})
            send_ms2_finished(self.email, self.results_link)
        except Exception as e:
            logger.critical('email error')
            #logger.critical("Error sending email: {}".format(e.message))
        logger.critical('email end function')
        
        self.query_progress = 0
        self.prepare_progress = 0

    def set_status(self, status, create = False):
        posts = self.mongo.posts
        time_stamp = datetime.utcnow()
        post_id = self.jobid + "_" + "status"
        if create:
            posts.update_one({'_id': post_id},{'$set': {'_id': post_id,
                                                        'date': time_stamp,
                                                        'n_masses': str(self.n_masses),
                                                        'progress': str(self.progress),
                                                        'status': status,
                                                        'error_info': ''}},
                             upsert=True)
        else:
            posts.update_one({'_id': post_id}, {'$set': {'_id': post_id,
                                                         'n_masses': str(self.n_masses),
                                                         'progress': str(self.progress),
                                                         'status': status}},
                             upsert=True)

    def update_progress(self):
        self.progress +=  1
        posts = self.mongo.posts
        post_id = self.jobid + "_" + "status"
        posts.update_one({'_id': post_id}, {'$set': {'_id': post_id,
                                                     'n_masses': str(self.n_masses),
                                                     'progress': str(self.progress)}},
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

    def mongo_save(self, file, step=""):
        to_save = file.to_json(orient='split')
        id = self.jobid + "_" + step
        self.gridfs.put(to_save, _id=id, encoding='utf-8', project_name = self.project_name)
