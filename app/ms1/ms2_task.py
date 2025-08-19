import pandas as pd
import numpy as np
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
import re
import psutil
from dask.diagnostics import ResourceProfiler

from dask.graph_manipulation import bind
from datetime import datetime
from dask.distributed import Client, LocalCluster, fire_and_forget, as_completed, get_client, wait
from django.urls import reverse
from .utilities import connect_to_mongoDB, connect_to_mongo_gridfs, ms2_api_search, fetch_ms2_files
from ..feature.feature import FeatureList, MS2_Spectrum
from ..feature.score_algo import SpectraScorer
from ...tools.ms2.file_manager import MS2_Parser
from ...tools.ms2.send_email import send_ms2_finished


# Set up logging
logger = logging.getLogger("nta_app.ms1")


class MS2Run:
    def __init__(
        self,
        input_df,
        ms1_chems_df,
        mode="pos",
        parameters=None,
        # mongo_address=None,
        # jobid="00000000",
    ):
        # Define from inputs
        self.mode = mode
        self.input_li = input_df
        self.ms1_chems = ms1_chems_df
        self.parameters = parameters
        self.ppm = parameters["ppm"]
        self.ms1_mass_cutoff = parameters["ms1_mass_cutoff"]
        self.ms1_RT_cutoff = parameters["ms1_RT_cutoff"]
        self.precursor_mass_accuracy = float(parameters["precursor_mass_accuracy"])
        self.fragment_mass_accuracy = float(parameters["fragment_mass_accuracy"])
        self.jobid = parameters["job_id"]
        # Define step/time/client
        self.step = "Started"
        self.time_log = {"step": [], "start": []}
        self.client = None
        # Define intermediary variables
        self.n_masses = 1
        self.progress = 0
        self.features = None
        self.cfmid_responses = []
        self.spectra_df = None
        # Define output variables
        self.ms2_out = None
        self.combined_out = None

    def execute(self):
        # Get memory usage at onset
        self.log_memory_usage("Start")
        # Check for ms1_chems
        if self.ms1_chems is not None:
            # self.set_status("Filtering MS2 features")
            self.filter_features()
            self.log_memory_usage("Filtering MS2 features")
            self.log_dask_memory("Filtering MS2 features")
        # Build feature list from input_dict
        # self.set_status("Extracting Spectra Data")
        self.construct_featurelist()
        self.log_memory_usage("Extracting Spectra Data")
        self.log_dask_memory("Extracting Spectra Data")
        # Retrieve CFMID spectra from API
        # self.set_status("Retrieving Reference Spectra")
        self.get_CFMID_spectra()
        self.log_memory_usage("Retrieving Reference Spectra")
        self.log_dask_memory("Retrieving Reference Spectra")
        # Save spectra information (?)
        # self.set_status("Saving Spectral Info")
        self.save_spectral_info()
        self.log_memory_usage("Saving Spectral Info")
        self.log_dask_memory("Saving Spectral Info")
        # Calculate spectral similarity scores
        # self.set_status("Calculating Similarity Scores")
        self.calc_CFMID_similarity()
        self.log_memory_usage("Calculating Similarity Scores")
        self.log_dask_memory("Calculating Similarity Scores")
        # self.set_status("Saving Data")
        self.save_data()
        self.log_memory_usage("Saving Data")
        self.log_dask_memory("Saving Data")
        logger.critical("Run Finished")
        logger.info(self.report_time_logs())
        logger.warn("MS2 job {}: Processing complete.".format(self.jobid))

    def log_memory_usage(self, step_name):
        """Logs the current memory usage."""
        process = psutil.Process()
        mem_info = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
        logger.info("[Job ID: %s] Memory usage after %s: %.2f MB", self.jobid, step_name, mem_info)

    def log_dask_memory(self, step_name):
        # Logs memory usage from all workers in a Dask cluster.
        if not self.client:
            return  # Skip if Dask client is not set
        try:
            worker_memory = self.client.run(lambda: psutil.Process().memory_info().rss / (1024 * 1024))
            logger.info("[Job ID: %s] Memory after %s - %s", self.jobid, step_name, worker_memory)
        except Exception as e:
            logger.error("Failed to log Dask worker memory: %s", repr(e))

    def calc_mono_mass(self):
        """
        Take input list, and iterate through items, adding "MONOMASS" to dict
        by correcting the "MASS" key:value based on the "CHARGE" key:value.

        Returns dataframe of input_li
        """
        # Get inputs
        input_li = self.input_li.copy()
        mode = self.mode
        # Define proton
        proton = 1.0073
        # Iterate
        for item in input_li:
            # Check "CHARGE"
            if item["CHARGE"] is not None:
                # Regex item["CHARGE"] to determine charge number (i.e., level)
                re_pattern = "\d+"
                level = int(re.search(re_pattern, item["CHARGE"]).group())
                # Calculate correction
                correction = level * proton
                # Check mode
                if mode == "pos":
                    # Calculate "MONOMASS"
                    item["MONOMASS"] = round(item["MASS"] - correction, 6)
                else:
                    item["MONOMASS"] = round(item["MASS"] + correction, 6)
            else:
                item["MONOMASS"] = round(item["MASS"], 6)
        # Convert to df
        output = pd.DataFrame(input_li)
        # Return output
        return output

    def filter_features(self):
        """
        If MS1 data is present, filter MS2 features for scoring based on whether or not
        the feature is present in the MS1 dataframe.
        """
        # Get inputs
        ms1_chems = self.ms1_chems.copy()
        # Group ms1_chems by Feature ID, Mass, RT
        ms1_mrts = ms1_chems.groupby(["Mass", "Retention Time", "Feature ID"])["DTXSID"].apply(list).reset_index()
        # Get rounded columns
        ms1_mrts["Rounded Mass"] = ms1_mrts["Mass"].round(2)
        ms1_mrts["Rounded Retention Time"] = ms1_mrts["Retention Time"].round(1)
        # Get input_li as df
        input_df = self.calc_mono_mass()
        cols = input_df.columns.tolist()
        # Get rounded columns
        input_df["Rounded Mass"] = input_df["MONOMASS"].round(2)
        input_df["Rounded Retention Time"] = input_df["RT"].round(1)
        # do merge on Rounded Mass and Rounded Retention_Time
        inner = pd.merge(ms1_mrts, input_df, how="inner", on=["Rounded Mass", "Rounded Retention Time"])
        # Set ppm
        ppm = True
        # Assess MS1 and MS2 Mass and RT thresholds
        mass_cutoff = self.ms1_mass_cutoff
        RT_cutoff = self.ms1_RT_cutoff
        # Mass and RT diffs
        inner["Mass diff"] = abs(inner["Mass"] - inner["MONOMASS"])
        inner["RT diff"] = abs(inner["Retention Time"] - inner["RT"])
        # ppm adjustment
        if ppm:
            inner["Mass diff"] = (inner["Mass diff"] / inner["Mass"]) * 10**6
        # Match thresholds
        inner["Mass Match?"] = np.where(inner["Mass diff"] < mass_cutoff, 1, 0)
        inner["RT Match?"] = np.where(inner["RT diff"] < RT_cutoff, 1, 0)
        # Subset by matching columns
        filtered_features = inner.loc[((inner["Mass Match?"] == 1) & (inner["RT Match?"] == 1)), :]
        # Convert filtered df back to list of dicts
        filtered_features = filtered_features[cols].to_dict("records")
        # Store in class attribute
        self.input_li = filtered_features

    def construct_featurelist(self):
        """
        Prepares FeatureList object from the input dict
        """
        # Get inputs
        input_li = self.input_li.copy()
        mode = self.mode
        # Get # of mass features
        self.n_masses = len(self.input_li)
        # Print to logger
        logger.info("Total number of features: {}".format(self.n_masses))
        # Create feature list
        tmp_feature_list = FeatureList()
        # pass list of dicts to 'update_feature_list'
        tmp_feature_list.update_feature_list(input_li, POSMODE=mode == "pos")
        # Save feature_list to self.features
        self.features = tmp_feature_list

    def get_CFMID_spectra(self):
        """
        Instantiate pos_list and neg_list with tuples of unique masses in the FeatureList and corrsponding mode. Iterate through list
        to get CFMID data. Returned spectra are appended to corresponding Features in the feature list using mass to join spectra.
        """
        # Get features list, add mode
        all_masses = (
            [(mass, "ESI-MSMS-" + str(self.mode)) for mass in self.features.get_masses(neutral=True)]
            if len(self.features) > 0
            else []
        )
        # Print len masses to logger
        self.n_masses = len(all_masses)
        logger.info(f"Number of features in list: {self.n_masses}")
        # Define chunk size
        chunk_size = 100
        # Initialize timer start
        start = time.perf_counter()
        # Check length of masses
        if self.n_masses > 0:
            # Set cfmid_responses to blank list
            self.cfmid_responses = []
            # Proceed through loop
            for idx in range(0, self.n_masses, chunk_size):
                chunk = all_masses[idx : min(idx + chunk_size, self.n_masses)]
                batch_results = []
                logger.info(f"API search: {chunk_size} of {len(all_masses)} structures")
                logger.info(f"\t\t\t Total count: {idx}")
                asyncio.run(ms2_api_search(batch_results, chunk, self.precursor_mass_accuracy, self.jobid))
                self.cfmid_responses.extend(batch_results)
            logger.info(f"API search time: {time.perf_counter() - start} for {len(all_masses)} structures")
        else:
            logger.warning("No masses to process.")

    # NTAW-795 Add spectral information into MS2 workflow results
    def save_spectral_info(self):
        # filter out entries with no spectra data, and remove mass and mode information
        responses = [item["data"] for item in self.cfmid_responses if item.get("data") is not None]
        # Merge the list of DTXCID-spectra dictionaries into a single dictionary
        new_list = []
        for dict in responses:
            new_dict = {k[0]: v for k, v, in dict.items()}
            new_list.append(new_dict)
        spectra_dict = {}
        for d in new_list:
            spectra_dict.update(d)

        # Convert the spectra dataframes into arrays of two-item arrays
        processed_spectra_dict = {}
        for key, inner_dict in spectra_dict.items():
            processed_inner = {}
            for item_key, df in inner_dict.items():
                # Replace df with df.spectrum_df copy
                temp_specta_list = df.spectrum_df.copy()[["FRAGMENT_MASS", "INTENSITY"]].values.tolist()
                temp_specta_list.sort(key=lambda x: x[0])
                processed_inner[item_key] = temp_specta_list
            processed_spectra_dict[key] = processed_inner
        # Convert the spectra_dict into a dataframe holding the energy0, energy1, and energy2 spectral data for each unique DTXCID
        spectra_df = pd.DataFrame.from_dict(processed_spectra_dict, orient="index").reset_index()
        spectra_df.columns = ["DTXCID", "energy0", "energy1", "energy2"]
        self.spectra_df = spectra_df

    def calc_CFMID_similarity(self):
        """
        For each feature that corresponds to the input mass in the Feature List
        object then add a cfmid result.

        :param cfmid_response: nested dict returned from processing the results of a cfmid query
        :type cfmid_response: dict

        """
        dask_scheduler = os.environ.get("DASK_SCHEDULER")
        dask_client = Client(dask_scheduler)

        ### This version works, but not as efficient as queing all tasks
        ###
        ###

        for idx, cfmid_response in enumerate(self.cfmid_responses):
            if cfmid_response["data"] is None:
                logger.info(f'Found 0 structures for mass {cfmid_response["mass"]}')
                continue
            # Logger statements
            logger.info(f'Found {len(cfmid_response["data"])} structures for mass {cfmid_response["mass"]}')
            logger.info(f"\t\t\t Total Progress: {idx + 1} / {len(self.cfmid_responses)} structures")
            # Get features from cfmid_response
            matched_features = self.features.get_features(cfmid_response["mass"], by="neutral_mass")
            # Pass to dask_client
            scattered_data = dask_client.scatter(cfmid_response["data"])
            # Instantiate task_list and feature_list
            task_list = []
            feature_list = []
            # Iterate through features, calculate similarity
            for feature in matched_features:
                task_list.append(dask_client.submit(feature.dask_calc_similarity, scattered_data))
                feature_list.append(feature)
            # Gather results, save results to feature attribute
            results = dask_client.gather(task_list)
            for feature, result in zip(feature_list, results):
                feature.reference_scores = result

    def save_data(self):
        # log self
        inputParameters = self.parameters
        logger.info("save_data - inputParameters:")
        logger.info(inputParameters)
        # Delete csrfmiddlewaretoken from inputParameters
        del inputParameters["csrfmiddlewaretoken"]
        # convert inputParameters to a dataframe and re-index the dataframe so it is no longer indexed by the dictionary keys
        inputParameters_df = pd.DataFrame.from_dict(inputParameters, orient="index").reset_index().drop(columns="index")
        # Add column headers to inputParameters_df
        inputParameters_df.columns = ["Parameter", "Value"]
        # log inputParameters_df
        logger.info("save_data - inputParameters_df:")
        logger.info(inputParameters_df)
        # Convert features to df
        df = self.features.to_df().sort_values(by=["ID", "Q-SCORE"], ascending=[True, False], ignore_index=True)
        # Merge spectrum data onto CFMID results data frames
        df_combined = pd.merge(df, self.spectra_df, on="DTXCID", how="left")
        # Save df_combined to combined_out
        self.combined_out = df_combined

    def log_time(self):
        self.time_log["start"].append(time.perf_counter())
        self.time_log["step"].append(self.get_step())

    def report_time_logs(self):
        total_time = max(self.time_log["start"]) - min(self.time_log["start"])
        step_time = {}
        for idx, step in enumerate(self.time_log["step"][:-1]):
            step_time[step] = self.time_log["start"][idx + 1] - self.time_log["start"][idx]
        return f"Total run time: {total_time} \n {json.dumps(step_time, indent = 6)}"

    def get_step(self):
        return self.step
