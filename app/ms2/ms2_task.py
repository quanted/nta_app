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
from .ms2_functions import compare_mgf_df
from ...tools.ms2.send_email import send_ms2_finished

NO_DASK = False  # set this to True to run locally without dask (for debug purposes)

logger = logging.getLogger("nta_app.ms2")
logger.setLevel(logging.INFO)


def run_ms2_dask(parameters, input_dfs, jobid = "00000000", verbose = True):
    in_docker = os.environ.get("IN_DOCKER") != "False"
    mongo_address = os.environ.get('MONGO_SERVER')
    link_address = reverse('ms2_results', kwargs={'jobid': self.jobid})
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
        self.project_name = parameters['project_name']
        self.input_dfs = input_dfs
        self.results_dfs = [[None],[None]]
        self.email = parameters['results_email']
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
        self.set_status('Processing', create = True)
        self.results_dfs[0] = pd.concat([compare_mgf_df(x, self.precursor_mass_accuracy, self.fragment_mass_accuracy, POSMODE=True) for x in self.input_dfs[0]])
        self.results_dfs[1] = pd.concat([compare_mgf_df(x, self.precursor_mass_accuracy, self.fragment_mass_accuracy, POSMODE=False) for x in self.input_dfs[1]])
        self.mongo_save(self.results_dfs[0], step=FILENAMES['final_output'][0])
        self.mongo_save(self.results_dfs[1], step=FILENAMES['final_output'][1])
        self.set_status('Completed')
        self.send_email()
        logger.critical('Run Finished')

    def send_email(self):
        try:
            #link_address = reverse('ms2_results', kwargs={'jobid': self.jobid})
            send_ms2_finished(self.email, self.results_link)
        except Exception as e:
            logger.critical('email error')
            logger.critical("Error sending email: {}".format(e.message))
        logger.critical('email end function')


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

    def mongo_save(self, file, step=""):
        to_save = file.to_json(orient='split')
        id = self.jobid + "_" + step
        self.gridfs.put(to_save, _id=id, encoding='utf-8', project_name = self.project_name)
