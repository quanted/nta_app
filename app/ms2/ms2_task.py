import pandas as pd
import dask
import os
import csv
import time
import logging
import traceback
import shutil
import json
import asyncio
import io
import psutil
from dask.distributed import get_task_stream

from dask.graph_manipulation import bind
from datetime import datetime
from dask.distributed import Client, LocalCluster, fire_and_forget, as_completed, get_client, wait
from django.urls import reverse
from .utilities import connect_to_mongoDB, connect_to_mongo_gridfs, ms2_api_search, fetch_ms2_files
from ..feature.feature import FeatureList, MS2_Spectrum
from ..feature.score_algo import SpectraScorer
from ...tools.ms2.file_manager import MS2_Parser
from ...tools.ms2.send_email import send_ms2_finished


NO_DASK = False  # set this to True to run locally without dask (for debug purposes)

# Set up logging
logger = logging.getLogger("nta_app.ms2")


def run_ms2_dask(parameters, jobid="00000000", verbose=True):
    # log parameters
    logger.info("Parameters:")
    logger.info(parameters)
    logger.info("Job ID: {}".format(jobid))

    in_docker = os.environ.get("IN_DOCKER") != "False"
    mongo_address = os.environ.get("MONGO_SERVER")
    link_address = reverse("ms2_results", kwargs={"jobid": jobid})
    if NO_DASK:
        run_ms2(parameters, mongo_address, jobid, results_link=link_address, verbose=verbose, in_docker=in_docker)
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
        logger.info("Dask URL: {dask_client.dashboard_link}")
    # dask_input_dfs = dask_client.scatter(input_dfs)
    logger.info("Submitting Nta ms2 Dask task")
    task = dask_client.submit(
        run_ms2,
        parameters,
        mongo_address,
        jobid,
        results_link=link_address,
        verbose=verbose,
        in_docker=in_docker,
    )  # dask_input_dfs, mongo_address, jobid, results_link=link_address,verbose=verbose, in_docker=in_docker)
    fire_and_forget(task)


def run_ms2(parameters, mongo_address=None, jobid="00000000", results_link="", verbose=True, in_docker=True):
    ms2_run = MS2Run(parameters, mongo_address, jobid, results_link, verbose, in_docker=in_docker)
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


FILENAMES = {"final_output": ["CFMID_results_pos", "CFMID_results_neg", "input_parameters"]}


class MS2Run:
    def __init__(
        self,
        parameters=None,
        mongo_address=None,
        jobid="00000000",
        results_link=None,
        verbose=True,
        in_docker=True,
    ):
        logger.info("[Job ID: {}] MS2Run initialize - started".format(jobid))
        self.inputParameters = parameters["inputParameters"]
        self.project_name = parameters["project_name"]
        self.n_masses = 1
        self.progress = 0
        self.input_dfs = {"pos": [None], "neg": [None]}
        self.features = {"pos": [], "neg": []}
        self.cfmid_responses = []
        # self.email = parameters['results_email']
        self.results_link = results_link
        self.precursor_mass_accuracy = float(parameters["precursor_mass_accuracy"])
        self.fragment_mass_accuracy = float(parameters["fragment_mass_accuracy"])
        self.jobid = jobid
        self.verbose = verbose
        self.in_docker = in_docker
        self.mongo_address = mongo_address
        self.mongo = connect_to_mongoDB(self.mongo_address)
        self.gridfs = connect_to_mongo_gridfs(self.mongo_address)
        self.step = "Started"  # tracks the current step (for fail messages)
        self.time_log = {"step": [], "start": []}

    def execute(self):
        with get_task_stream() as task_profiler:
            logger.info(f"Task Stream memory usage: {task_profiler.data}")
            self.set_status("Parsing MS2 Data", create=True)
            self.parse_uploaded_files()
            logger.info(f"Task Stream memory usage: {task_profiler.data}")

            self.set_status("Extracting Spectra Data")
            self.construct_featurelist()
            logger.info(f"Task Stream memory usage: {task_profiler.data}")

            self.set_status("Retrieving Reference Spectra")
            self.get_CFMID_spectra()
            logger.info(f"Task Stream memory usage: {task_profiler.data}")

            self.set_status("Calculating Similarity Scores")
            self.calc_CFMID_similarity()
            logger.info(f"Task Stream memory usage: {task_profiler.data}")

            self.set_status("Saving Data")
            self.save_data()
            logger.info(f"Task Stream memory usage: {task_profiler.data}")
        # self.set_status("Parsing MS2 Data", create=True)
        # self.parse_uploaded_files()

        # self.set_status("Extracting Spectra Data")
        # self.construct_featurelist()

        # self.set_status("Retrieving Reference Spectra")
        # self.get_CFMID_spectra()

        # self.set_status("Calculating Similarity Scores")
        # self.calc_CFMID_similarity()

        # self.set_status("Saving Data")
        # self.save_data()

        self.set_status("Completed")
        # self.send_email()
        logger.critical("Run Finished")
        logger.info(self.report_time_logs())
        logger.warn("MS2 job {}: Processing complete.".format(self.jobid))

    def parse_uploaded_files(self):
        """
        Parse MGF file stored on mongo DB. Uses jobID to fetch related MS2 files before parse
        """
        grid_out = fetch_ms2_files(self.jobid)
        for file in grid_out:
            file_index = grid_out.index(file)
            filename = self.inputParameters["fileUpload"][1][file_index]
            if file.mode == "neg":
                self.input_dfs["neg"].append(MS2_Parser.parse_file(file, filename))
            else:
                self.input_dfs["pos"].append(MS2_Parser.parse_file(file, filename))

    def construct_featurelist(self):
        """
        Prepares pos- and neg-mode FeatureList object from the input dfs
        """
        self.n_masses = len(self.input_dfs["neg"]) + len(self.input_dfs["pos"])

        logger.info("Total number of features: {}".format(self.n_masses))

        for mode, df_list in self.input_dfs.items():
            tmp_feature_list = FeatureList()
            for data_block in df_list:
                tmp_feature_list.update_feature_list(data_block, POSMODE=mode == "pos")
                self.update_progress()
            self.features[mode] = tmp_feature_list

    def get_CFMID_spectra(self):
        """
        Instantiate pos_list and neg_list with tuples of unique masses in the FeatureList and corrsponding mode. Iterate through list
        to get CFMID data. Returned spectra are appended to corresponding Features in the feature list using mass to join spectra.
        """
        self.reset_progress()
        pos_list = (
            [(mass, "ESI-MSMS-pos") for mass in self.features["pos"].get_masses(neutral=True)]
            if len(self.features["pos"]) > 0
            else []
        )
        neg_list = (
            [(mass, "ESI-MSMS-neg") for mass in self.features["neg"].get_masses(neutral=True)]
            if len(self.features["neg"]) > 0
            else []
        )
        all_masses = pos_list + neg_list
        self.n_masses = len(all_masses)
        logger.info(f"Number of features in list: {self.n_masses}")
        chunk_size = 100
        start = time.perf_counter()  # Initialize start before the loop

        if self.n_masses > 0:
            for idx in range(0, self.n_masses, chunk_size):
                chunk = all_masses[idx : min(idx + chunk_size, self.n_masses)]
                self.cfmid_responses = []
                logger.info(f"API search: {chunk_size} of {len(all_masses)} structures")
                logger.info(f"\t\t\t Total count: {idx}")
                asyncio.run(ms2_api_search(self.cfmid_responses, chunk, self.precursor_mass_accuracy, self.jobid))

            logger.info(f"API search time: {time.perf_counter() - start} for {len(all_masses)} structures")
        else:
            logger.warning("No masses to process.")

    def calc_CFMID_similarity(self):
        """
        For each feature that corresponds to the input mass in the Feature List
        object then add a cfmid result.

        :param cfmid_response: nested dict returned from processing the results of a cfmid query
        :type cfmid_response: dict

        """
        self.reset_progress()
        dask_scheduler = os.environ.get("DASK_SCHEDULER")
        dask_client = Client(dask_scheduler)

        ### This version works, but not as efficient as queing all tasks
        ###
        ###

        for idx, cfmid_response in enumerate(self.cfmid_responses):
            if cfmid_response["data"] is None:
                logger.info(f'Found 0 structures for mass {cfmid_response["mass"]}')
                self.update_progress()
                continue

            logger.info(f'Found {len(cfmid_response["data"])} structures for mass {cfmid_response["mass"]}')
            logger.info(f"\t\t\t Total Progress: {idx + 1} / {len(self.cfmid_responses)} structures")

            feature_mode = "pos" if cfmid_response["mode"] == "ESI-MSMS-pos" else "neg"
            matched_features = self.features[feature_mode].get_features(cfmid_response["mass"], by="neutral_mass")

            scattered_data = dask_client.scatter(cfmid_response["data"])

            task_list = []
            feature_list = []

            for feature in matched_features:
                task_list.append(dask_client.submit(feature.dask_calc_similarity, scattered_data))
                feature_list.append(feature)

            results = dask_client.gather(task_list)
            for feature, result in zip(feature_list, results):
                feature.reference_scores = result
            self.update_progress()

        ### This version crashes in the final for loop -> concurrent.futures._base.CancelledError
        ### Tasks are being canceled at some point during the run, but unclear what the root cause is
        ###

        # feature_list = []
        # task_list = []

        # for idx, cfmid_response in enumerate(self.cfmid_responses):
        #     if cfmid_response['data'] is None:
        #         logger.info(f'Found 0 structures for mass {cfmid_response["mass"]}')
        #         logger.info(f'\t\t\t Total Progress: {idx + 1} / {len(self.cfmid_responses)} structures')
        #         self.update_progress()
        #         continue

        #     logger.info(f'Found {len(cfmid_response["data"])} structures for mass {cfmid_response["mass"]}')
        #     logger.info(f'\t\t\t Total Progress: {idx + 1} / {len(self.cfmid_responses)} structures')

        #     feature_mode = 'pos' if cfmid_response['mode'] == 'ESI-MSMS-pos' else 'neg'
        #     matched_features = self.features[feature_mode].get_features(cfmid_response['mass'], by='neutral_mass')

        #     scattered_data = dask_client.scatter(cfmid_response['data'])

        #     for feature in matched_features:
        #         task_list.append(dask_client.submit(feature.dask_calc_similarity, scattered_data))
        #         feature_list.append(feature)

        # #results = dask_client.gather(task_list)
        # for feature, task in zip(feature_list, task_list):
        #     try:
        #         result = task.result()
        #     except Exception as e:
        #         logger.info(f"Failed on feature: {feature}, {task}")
        #         logger.info(f"Exception: {e}")
        #         task_retry = dask_client.retry(task)        #This throws an error, taks are being cancelled at some point. Needs additional troubleshooting
        #         result = task_retry.result()
        #     feature.reference_scores = result
        #     self.update_progress()

    def save_data(self):
        # log self
        inputParameters = self.inputParameters
        logger.info("save_data - inputParameters:")
        logger.info(inputParameters)
        # convert inputParameters to a dataframe
        inputParameters_df = pd.DataFrame.from_dict(inputParameters, orient="index")
        # log inputParameters_df
        logger.info("save_data - inputParameters_df:")
        logger.info(inputParameters_df)

        # self.mongo_save(self.features['neg'].to_df().sort_values(by = ['ID', 'Q-SCORE'], ascending = [True, False], ignore_index = True), step=FILENAMES['final_output'][0])
        # self.mongo_save(self.features['pos'].to_df(), step=FILENAMES['final_output'][1])
        # 2/23/2023 Reverse the filenames index, currently pointing to the wrong file
        self.mongo_save(
            self.features["neg"].to_df().sort_values(by=["ID", "Q-SCORE"], ascending=[True, False], ignore_index=True),
            step=FILENAMES["final_output"][1],
        )
        # self.mongo_save(self.features["pos"].to_df(), step=FILENAMES["final_output"][0])
        self.mongo_save(
            self.features["pos"].to_df().sort_values(by=["ID", "Q-SCORE"], ascending=[True, False], ignore_index=True),
            step=FILENAMES["final_output"][0],
        )
        self.mongo_save(inputParameters_df, step=FILENAMES["final_output"][2])

    def send_email(self):
        try:
            # link_address = reverse('ms2_results', kwargs={'jobid': self.jobid})
            send_ms2_finished(self.email, self.results_link)
        except Exception as e:
            logger.critical("email error")
            # logger.critical("Error sending email: {}".format(e.message))
        logger.critical("email end function")

        self.query_progress = 0
        self.prepare_progress = 0

    def set_status(self, status, create=False):
        logger.info(
            f"\n============= Job ID: {self.jobid} \n============= Starting Process: \n============= {status}  \n============="
        )
        self.step = status
        self.log_time()
        posts = self.mongo.posts
        time_stamp = datetime.utcnow()
        post_id = self.jobid + "_" + "status"
        if create:
            posts.update_one(
                {"_id": post_id},
                {
                    "$set": {
                        "_id": post_id,
                        "date": time_stamp,
                        "n_masses": str(self.n_masses),
                        "progress": str(self.progress),
                        "status": status,
                        "error_info": "",
                    }
                },
                upsert=True,
            )
        else:
            posts.update_one(
                {"_id": post_id},
                {
                    "$set": {
                        "_id": post_id,
                        "n_masses": str(self.n_masses),
                        "progress": str(self.progress),
                        "status": status,
                    }
                },
                upsert=True,
            )

    def log_time(self):
        self.time_log["start"].append(time.perf_counter())
        self.time_log["step"].append(self.get_step())

    def report_time_logs(self):
        total_time = max(self.time_log["start"]) - min(self.time_log["start"])
        step_time = {}
        for idx, step in enumerate(self.time_log["step"][:-1]):
            step_time[step] = self.time_log["start"][idx + 1] - self.time_log["start"][idx]
        return f"Total run time: {total_time} \n {json.dumps(step_time, indent = 6)}"

    def reset_progress(self, progress_value=0):
        self.progress = progress_value
        posts = self.mongo.posts
        post_id = self.jobid + "_" + "status"
        posts.update_one(
            {"_id": post_id},
            {"$set": {"_id": post_id, "n_masses": str(self.n_masses), "progress": str(self.progress)}},
            upsert=True,
        )

    def update_progress(self, step_size=1):
        self.progress += step_size
        posts = self.mongo.posts
        post_id = self.jobid + "_" + "status"
        posts.update_one(
            {"_id": post_id},
            {"$set": {"_id": post_id, "n_masses": str(self.n_masses), "progress": str(self.progress)}},
            upsert=True,
        )

    def set_except_message(self, e):
        posts = self.mongo.posts
        time_stamp = datetime.utcnow()
        post_id = self.jobid + "_" + "status"
        posts.update_one({"_id": post_id}, {"$set": {"_id": post_id, "date": time_stamp, "error_info": e}}, upsert=True)

    def get_step(self):
        return self.step

    def mongo_save(self, file, step=""):
        to_save = file.to_json(orient="split")
        id = self.jobid + "_" + step
        self.gridfs.put(to_save, _id=id, encoding="utf-8", project_name=self.project_name)
