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
from .merge_functions import process_MS2_data
from ...tools.ms2.send_email import send_ms2_finished

NO_DASK = False  # set this to True to run locally without dask (for debug purposes)

logger = logging.getLogger("nta_app.merge")


def run_merge_dask(parameters, input_dfs, jobid="00000000", verbose=True):
    in_docker = os.environ.get("IN_DOCKER") != "False"
    mongo_address = os.environ.get("MONGO_SERVER")
    link_address = reverse("merge_results", kwargs={"jobid": jobid})
    if NO_DASK:
        run_merge(
            parameters, input_dfs, mongo_address, jobid, results_link=link_address, verbose=verbose, in_docker=in_docker
        )
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
    # dask_input_dfs = dask_client.scatter(input_dfs)
    logger.info("Submitting Nta merge Dask task")
    task = dask_client.submit(
        run_merge,
        parameters,
        input_dfs,
        mongo_address,
        jobid,
        results_link=link_address,
        verbose=verbose,
        in_docker=in_docker,
    )
    fire_and_forget(task)


def run_merge(
    parameters, input_dfs, mongo_address=None, jobid="00000000", results_link="", verbose=True, in_docker=True
):
    merge_run = MergeRun(parameters, input_dfs, mongo_address, jobid, results_link, verbose, in_docker=in_docker)
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


class MergeRun:
    def __init__(
        self,
        parameters=None,
        input_data=None,
        mongo_address=None,
        jobid="00000000",
        results_link=None,
        verbose=True,
        in_docker=True,
    ):
        self.project_name = parameters["project_name"]
        logger.info(f"\n============= Job ID: {jobid}")
        logger.info(input_data)
        self.input_ms1 = input_data["MS1"]
        # self.input_ms2 = [input_data["MS2_pos"], input_data["MS2_neg"]] # NTAW-158: Update how MS2 data is handled, needs to be a list of MS2 files / ignore this
        self.input_ms2 = input_data["MS2_pos"] + input_data["MS2_neg"]
        self.n_files = len(self.input_ms2)
        self.results_df = [None]
        self.results_link = results_link
        self.mass_accuracy_tolerance = float(parameters["mass_accuracy_tolerance"])
        self.rt_tolerance = float(parameters["rt_tolerance"])
        self.jobid = jobid
        self.verbose = verbose
        self.in_docker = in_docker
        self.mongo_address = mongo_address
        self.mongo = connect_to_mongoDB(self.mongo_address)
        self.gridfs = connect_to_mongo_gridfs(self.mongo_address)
        self.step = "Started"  # tracks the current step (for fail messages)
        # NTAW-158: Adjust sheet names pulled from MS1 results
        self.ms1_data_map = (
            {"chemical_results": self.input_ms1} if isinstance(self.input_ms1, pd.DataFrame) else self.input_ms1
        )
        # self.ms1_data_map = (
        #     {"dsstox_search": self.input_ms1} if isinstance(self.input_ms1, pd.DataFrame) else self.input_ms1
        # )

        logger.info(f"\n============= Job ID: {jobid}")

    def execute(self):
        self.set_status("Processing", create=True)
        if self.n_files > 0:
            # NTAW-158: Debugger statements
            logger.info("self.input_MS2:")
            logger.info(len(self.input_ms2))
            logger.info(self.input_ms2)
            logger.info("shape file_df:")
            logger.info(self.input_ms2[0]["file_df"].shape)
            logger.info("file_df:")
            logger.info(self.input_ms2[0]["file_df"])
            # NTAW-158: Adjust sheet names pulled from MS1 results
            self.ms1_data_map["chemical_results"] = process_MS2_data(
                self.input_ms1, self.input_ms2, self.mass_accuracy_tolerance, self.rt_tolerance
            )
            # self.ms1_data_map["dsstox_search"] = process_MS2_data(
            #     self.input_ms1, self.input_ms2, self.mass_accuracy_tolerance, self.rt_tolerance
            # )
            logger.info("Store file names")
            self.gridfs.put(
                "&&".join(self.ms1_data_map.keys()),
                _id=self.jobid + "_file_names",
                encoding="utf-8",
                project_name=self.project_name,
            )
            logger.info("Store data to each file name")
            for key in self.ms1_data_map.keys():
                self.mongo_save(self.ms1_data_map[key], data_name=key)
        self.set_status("Completed", progress=self.n_files)
        logger.info(f"[Job ID: {self.jobid}] Run Finished")

    def set_status(self, status, progress=0, create=False):
        self.step = status
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
                        "n_masses": str(self.n_files),
                        "progress": str(progress),
                        "status": status,
                        "error_info": "",
                    }
                },
                upsert=True,
            )
        else:
            posts.update_one(
                {"_id": post_id}, {"$set": {"_id": post_id, "progress": str(progress), "status": status}}, upsert=True
            )

    def set_except_message(self, e):
        posts = self.mongo.posts
        time_stamp = datetime.utcnow()
        post_id = self.jobid + "_" + "status"
        posts.update_one({"_id": post_id}, {"$set": {"_id": post_id, "date": time_stamp, "error_info": e}}, upsert=True)

    def get_step(self):
        return self.step

    def mongo_save(self, file, data_name=""):
        to_save = file.to_json(orient="split")
        id = self.jobid + "_" + data_name
        self.gridfs.put(to_save, _id=id, encoding="utf-8", project_name=self.project_name)
