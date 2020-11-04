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
from .utilities import connect_to_mongoDB, connect_to_mongo_gridfs
from .merge_functions import compare_mgf_df, count_masses
from ...tools.ms2.send_email import send_ms2_finished

NO_DASK = False  # set this to True to run locally without dask (for debug purposes)

logger = logging.getLogger("nta_app.merge")
logger.setLevel(logging.INFO)


def run_merge_dask(parameters, input_dfs, jobid = "00000000", verbose = True):
    in_docker = os.environ.get("IN_DOCKER") != "False"
    mongo_address = os.environ.get('MONGO_SERVER')
    link_address = reverse('merge_results', kwargs={'jobid': jobid})
    if NO_DASK:
        run_merge(parameters, input_dfs, mongo_address, jobid, results_link=link_address, verbose=verbose, in_docker=in_docker)
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
    logger.info("Submitting Nta merge Dask task")
    task = dask_client.submit(run_merge, parameters, dask_input_dfs, mongo_address, jobid, results_link=link_address,
                              verbose=verbose, in_docker=in_docker)
    fire_and_forget(task)


def run_merge(parameters, input_dfs,  mongo_address = None, jobid = "00000000", results_link = "", verbose = True,
            in_docker = True):
    merge_run = MergeRun(parameters, input_dfs, mongo_address, jobid, results_link, verbose, in_docker = in_docker)
    try:
        merge_run.execute()
    except Exception as e:
        trace = traceback.format_exc()
        logger.info(trace)
        fail_step = merge_run.get_step()
        merge_run.set_status("Failed on step: " + fail_step)
        error = repr(e)
        merge_run.set_except_message(error)
        raise e
    return True


FILENAMES = {'final_output': ['CFMID_results_pos', 'CFMID_results_neg']}


class MergeRun:
    
    def __init__(self, parameters=None, input_dfs=None, mongo_address = None, jobid = "00000000",
                 results_link = None, verbose = True, in_docker = True):
        self.project_name = parameters['project_name']
        self.input_dfs = input_dfs
        ##self.n_masses_pos = sum([count_masses(x, POSMODE=True) for x in self.input_dfs[0]])
        ##self.n_masses_neg = sum([count_masses(x, POSMODE=False) for x in self.input_dfs[1]])
        ##self.n_masses = self.n_masses_pos + self.n_masses_neg
        print("Total job number of masses: {}".format(self.n_masses))
        self.progress = 0
        self.results_dfs = [[None],[None]]
        #self.email = parameters['results_email']
        self.results_link = results_link
        ##self.precursor_mass_accuracy = float(parameters['precursor_mass_accuracy'])
        ##self.fragment_mass_accuracy = float(parameters['fragment_mass_accuracy'])
        self.mass_accuracy_tolerance = float(parameters['mass_accuracy_tolerance'])
        self.rt_tolerance = float(parameters['rt_tolerance'])
        self.jobid = jobid
        self.verbose = verbose
        self.in_docker = in_docker
        self.mongo_address = mongo_address
        self.mongo = connect_to_mongoDB(self.mongo_address)
        self.gridfs = connect_to_mongo_gridfs(self.mongo_address)
        self.step = "Started"  # tracks the current step (for fail messages)

    def execute(self):
        self.set_status('Processing', create = True)
        '''
        if len(self.input_dfs[0]) > 0:  # if there is at least one pos file
            self.results_dfs[0] = pd.concat([self.process_results(x, POSMODE=True) for x in self.input_dfs[0]])
            self.mongo_save(self.results_dfs[0], step=FILENAMES['final_output'][0])
        if len(self.input_dfs[1]) > 0:  # if there is at least one neg file
            self.results_dfs[1] = pd.concat([self.process_results(x, POSMODE=False) for x in self.input_dfs[1]])
            self.mongo_save(self.results_dfs[1], step=FILENAMES['final_output'][1])
        '''
        if len(self.input_dfs[0]) > 0:  # if there is at least one pos file
            self.results_dfs = pd.concat([self.process_results(x) for x in self.input_dfs)
            self.mongo_save(self.results_dfs, step=FILENAMES['final_output'])
        self.set_status('Completed', progress=self.n_masses)
        #self.send_email()
        logger.critical('Run Finished')
    '''
    def process_results(self, input_df, POSMODE=True):
        result = compare_mgf_df(input_df, self.precursor_mass_accuracy, self.fragment_mass_accuracy, POSMODE=POSMODE,
                                mongo=self.mongo, jobid= self.jobid, progress=self.progress)
        if POSMODE:
            self.progress = self.progress + self.n_masses_pos
        #self.set_status('Processing', progress=progress)
        return result
    '''
    def process_results(self, input_df):
        '''
        result = compare_mgf_df(input_df, self.precursor_mass_accuracy, self.fragment_mass_accuracy, POSMODE=POSMODE,
                                mongo=self.mongo, jobid= self.jobid, progress=self.progress)
        if POSMODE:
            self.progress = self.progress + self.n_masses_pos
        '''
        result = Process_matches(input_df, self.rt_tolerance, self.mass_accuracy_tolerance,
                        mongo=self.mongo, jobid= self.jobid, progress=self.progress)
        #self.set_status('Processing', progress=progress)
        return result

    def send_email(self):
        try:
            #link_address = reverse('ms2_results', kwargs={'jobid': self.jobid})
            send_ms2_finished(self.email, self.results_link)
        except Exception as e:
            logger.critical('email error')
            #logger.critical("Error sending email: {}".format(e.message))
        logger.critical('email end function')


    def set_status(self, status, progress=0, create = False):
        posts = self.mongo.posts
        time_stamp = datetime.utcnow()
        post_id = self.jobid + "_" + "status"
        if create:
            posts.update_one({'_id': post_id},{'$set': {'_id': post_id,
                                                        'date': time_stamp,
                                                        'n_masses': str(self.n_masses),
                                                        'progress': str(progress),
                                                        'status': status,
                                                        'error_info': ''}},
                             upsert=True)
        else:
            posts.update_one({'_id': post_id}, {'$set': {'_id': post_id,
                                                         'progress': str(progress),
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

    def mongo_save(self, file, step=""):
        to_save = file.to_json(orient='split')
        id = self.jobid + "_" + step
        self.gridfs.put(to_save, _id=id, encoding='utf-8', project_name = self.project_name)
