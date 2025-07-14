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
from zipfile import ZipFile, ZIP_DEFLATED
from openpyxl.utils import get_column_letter
import numpy as np
import gc
import psutil

# connect_to_mongoDB, connect_to_mongo_gridfs, reduced_file, api_search_masses, api_search_formulas,
from .utilities import *

from .heatmap import *
from .cv_scatterplot import *

from . import task_functions as task_fun
from .WebApp_plotter import WebApp_plotter
from .qNTA_class import qNTAClass
import io

logger = logging.getLogger("nta_app.ms1")

# import seaborn as sns
try:
    import seaborn as sns
except ModuleNotFoundError:
    logger.error("Seaborn is not installed. Please run 'pip install seaborn' to install it.")

# set this to true to run locally without test (for debug purposes)
NO_DASK = False


def run_nta_dask(
    parameters,
    input_dfs,
    tracer_df=None,
    run_sequence_pos_df=None,
    run_sequence_neg_df=None,
    qnta_df=None,
    val_df=None,
    jobid="00000000",
    verbose=True,
):
    in_docker = os.environ.get("IN_DOCKER") != "False"
    mongo_address = os.environ.get("MONGO_SERVER")
    logger.info("Mongo Address: {}".format(mongo_address))

    if NO_DASK:
        run_nta(
            parameters,
            input_dfs,
            tracer_df,
            run_sequence_pos_df,
            run_sequence_neg_df,
            qnta_df,
            val_df,
            mongo_address,
            jobid,
            verbose,
            in_docker=in_docker,
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

    input_dfs_size = len(input_dfs)
    logger.info("Before scatter input_dfs_size: {}".format(input_dfs_size))
    dask_input_dfs = dask_client.scatter(input_dfs)
    dask_input_dfs_size = len(dask_input_dfs)
    logger.info("After scatter dask_input_dfs_size: {}".format(dask_input_dfs_size))

    logger.warning("Submitting Nta Dask task")
    task = dask_client.submit(
        run_nta,
        parameters,
        dask_input_dfs,
        tracer_df,
        run_sequence_pos_df,
        run_sequence_neg_df,
        qnta_df,
        val_df,
        mongo_address,
        jobid,
        verbose,
        in_docker=in_docker,
    )
    fire_and_forget(task)


def run_nta(
    parameters,
    input_dfs,
    tracer_df=None,
    run_sequence_pos_df=None,
    run_sequence_neg_df=None,
    qnta_df=None,
    val_df=None,
    mongo_address=None,
    jobid="00000000",
    verbose=True,
    in_docker=True,
):
    nta_run = NtaRun(
        parameters,
        input_dfs,
        tracer_df,
        run_sequence_pos_df,
        run_sequence_neg_df,
        qnta_df,
        val_df,
        mongo_address,
        jobid,
        verbose,
        in_docker=in_docker,
    )
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
    def __init__(
        self,
        parameters=None,
        input_dfs=None,
        tracer_df=None,
        run_sequence_pos_df=None,
        run_sequence_neg_df=None,
        qnta_df=None,
        val_df=None,
        mongo_address=None,
        jobid="00000000",
        verbose=True,
        in_docker=True,
    ):
        logger.info(f"\n============= Job ID: {jobid}")
        logger.info("[Job ID: {}] Initializing NtaRun Task".format(jobid))
        logger.info("parameters= {}".format(parameters))
        self.parameters = parameters
        self.tracer_df = tracer_df
        self.tracer_dfs_out = None
        self.run_sequence_pos_df = run_sequence_pos_df
        self.run_sequence_neg_df = run_sequence_neg_df
        self.dfs = input_dfs
        self.qnta_df = qnta_df
        self.val_df = val_df
        self.qnta_dfs_out = None
        self.qnta_occ_input = None
        self.dfs_flagged = None  # DFs that will retain occurrences failing CV values
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
        self.base_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../.."))
        self.data_map = {}
        self.chem_res_map = {}
        self.qnta_map = {}
        self.tracer_map = {}
        self.occurrence_heatmap_map = {}
        self.cv_scatterplot_map = {}
        self.aq_plots_map = {}
        self.ecdf_plots_map = {}
        # self.data_dir = os.path.join(self.base_dir, 'data', self.jobid)
        # self.new_download_dir = os.path.join(self.data_dir, "new")
        self.step = "Started"  # tracks the current step (for fail messages)
        # os.mkdir(self.data_dir)
        # os.mkdir(self.new_download_dir)
        self.tracer_plots_out = []
        self.cc_plots_out = None
        self.occurrence_heatmaps_out = []
        self.cv_scatterplots_out = []
        self.aq_plots_out = []
        self.ecdf_plots_out = []

    def log_memory_usage(self, step_name):
        """Logs the current memory usage."""
        process = psutil.Process()
        mem_info = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
        logger.info("[Job ID: %s] Memory usage after %s: %.2f MB", self.jobid, step_name, mem_info)

    def execute(self):
        # Log the memory usage before executing MS1 run
        self.log_memory_usage("Start")

        # Work-around: Create filter_cv if it was not passed from parameters
        self.parameters.setdefault("filter_cv", "yes")

        # 0: create a run status in MongoDB
        self.set_status("Processing", create=True)
        # 0.9: Normalize all column names
        self.normalize_column_names(self.dfs)
        self.step = "Check for existence of required columns"
        # 1a: check existence of "Ionization mode" column
        self.check_existence_of_ionization_mode_column(self.dfs)
        # 1b: check existence of 'mass column'
        self.check_existence_of_mass_column(self.dfs)
        # 1c: check for alternate spellings of 'Retention_Time' column
        self.check_retention_time_column(self.dfs)
        # 1d: sort dataframe columns alphabetically
        self.dfs = [df.reindex(sorted(df.columns), axis=1) if df is not None else None for df in self.dfs]
        # 1e: create an analysis_parameters sheet
        self.create_analysis_parameters_sheet()
        # 1f: create run sequence sheets
        self.create_run_sequence_sheets()
        # 2: assign ids, separate passthrough cols, filter void volume, and flag duplicates
        self.step = "Flagging duplicates"
        self.assign_id()
        # NTAW-594
        self.all_headers, self.blank_headers, self.sample_headers = task_fun.get_sample_and_blank_headers(self.dfs)
        logger.info(f"all headers: {self.all_headers}")
        logger.info(f"blank headers: {self.blank_headers}")
        logger.info(f"sample headers: {self.sample_headers}")
        self.pass_through_cols()
        self.filter_void_volume(float(self.parameters["minimum_rt"][1]))  # throw out features below this (void volume)
        self.filter_duplicates()
        if self.verbose:
            logger.info("Flagged duplicates.")
            logger.info("dfs.size(): {}".format(len(self.dfs)))
            if self.dfs[0] is not None:
                logger.info("POS df length: {}".format(len(self.dfs[0])))
                # logger.info("POS df columns: {}".format(self.dfs[0].columns))
            if self.dfs[1] is not None:
                logger.info("NEG df length: {}".format(len(self.dfs[1])))
                # logger.info("NEG df columns: {}".format(self.dfs[1].columns))
        # 3a: statistics
        self.step = "Calculating statistics"
        self.calc_statistics()
        if self.verbose:
            logger.info("Calculated statistics.")
            if self.dfs[0] is not None:
                logger.info("POS df length: {}".format(len(self.dfs[0])))
            if self.dfs[1] is not None:
                logger.info("NEG df length: {}".format(len(self.dfs[1])))

        # 3b: Occurrence heatmap
        self.step = "Create heatmap"
        self.store_heatmap()

        # 4a: check tracers (optional)
        self.step = "Checking tracers"
        if self.verbose:
            # logger.info("Checking tracers: DF check for debugging.")
            # logger.info("dfs.size(): {}".format(len(self.dfs)))
            if self.dfs[0] is not None:
                logger.info("POS df length: {}".format(len(self.dfs[0])))
                # logger.info("POS df columns: {}".format(self.dfs[0].columns.tolist()))
            if self.dfs[1] is not None:
                logger.info("NEG df length: {}".format(len(self.dfs[1])))
                # logger.info("NEG df columns: {}".format(self.dfs[1].columns.tolist()))
        self.check_tracers()
        if self.verbose:
            logger.info("Checked tracers.")

        # 4b: CV Scatterplot
        self.step = "Create scatterplot"
        self.store_scatterplots()

        # Optional: Perform qNTA
        if self.parameters["do_qnta"][1] == "yes":
            self.step = "Performing qNTA"
            self.perform_qNTA()

        # 5a: clean features
        self.step = "Cleaning features"
        self.clean_features()

        if self.verbose:
            logger.info("Cleaned features.")
            if self.dfs[0] is not None:
                logger.info("POS df length: {}".format(len(self.dfs[0])))
            if self.dfs[1] is not None:
                logger.info("NEG df length: {}".format(len(self.dfs[1])))

        # 5b: Merge detection count columns onto tracers for export
        self.step = "Merge detection counts onto tracers"
        self.merge_columns_onto_tracers()

        if self.verbose:
            logger.info("Created flags.")
            if self.dfs[0] is not None:
                logger.info("POS df length: {}".format(len(self.dfs[0])))
                # NTAW-94 set flag for dashboard search
                self.dfs[0]["For_Dashboard_Search"] = "1"
                self.dfs_flagged[0]["For_Dashboard_Search"] = "1"
            if self.dfs[1] is not None:
                logger.info("NEG df length: {}".format(len(self.dfs[1])))
                # NTAW-94 set flag for dashboard search
                self.dfs[1]["For_Dashboard_Search"] = "1"
                self.dfs_flagged[1]["For_Dashboard_Search"] = "1"

        # 6: combine modes
        self.step = "Combining modes"
        self.combine_modes()
        if self.verbose:
            logger.info("Combined modes.")
            if self.df_combined is not None:
                logger.info("combined df length: {}".format(len(self.df_combined)))
            else:
                logger.info("combined df_flagged length: {}".format(len(self.df_flagged_combined)))

        # 7: search dashboard
        if self.parameters["search_dsstox"][1] == "yes":
            self.step = "Searching dsstox database"
            self.perform_dashboard_search()
            if self.parameters["search_hcd"][1] == "yes":
                self.step = "Searching Cheminformatics Hazard Module database"
                self.perform_hcd_search()

        # 8: Store excel data to MongoDB
        self.step = "Storing data"
        logger.info("Storing data into Mongo.")
        self.save_excel_to_mongo()
        logger.info("Storing decision tree info into Mongo.")
        self.save_decision_tree_info_to_mongo()

        # 9: set status to completed
        self.step = "Displaying results"
        self.set_status("Completed")
        self.log_memory_usage("Job Completed")
        logger.warning("MS1 job {}: Processing complete.".format(self.jobid))

    def normalize_column_names(self, input_dfs):
        """
        Normalize all column names in a list of DataFrames

        This function attempts to eliminate issues with the naming of columns in the input dataframes, including hidden spaces and encoding issues

        Args:
            self: The instance of the class (typically associated with object-oriented programming).
            input_dfs (list of pandas.DataFrame): A list of pandas DataFrames to check and modify.
        Returns:
            None: This function operates in place and modifies the input DataFrames.
        Example:
        """
        for df in input_dfs:
            if df is not None:
                df.columns = df.columns.astype(str)  # Ensure all column names are strings
                df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
                df.columns = df.columns.str.replace(r"\s+", " ", regex=True)  # Normalize spaces
                df.columns = df.columns.str.replace("\xa0", " ")  # Remove non-breaking spaces
        return

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
        # Iterate through list of dfs
        for df in input_dfs:
            if df is not None:
                if "Ionization_Mode" not in df.columns:
                    # create a new column with the header of "Ionization_Mode" and values of ionizationMode
                    df["Ionization_Mode"] = ionizationMode
            # the first element of input_dfs is the negative mode dataframe
            ionizationMode = "Esi-"
        return

    def check_existence_of_mass_column(self, input_dfs):
        """
        Check the existence of a 'Mass' or 'm/z' column in input dataframes and handle it accordingly.
        This function checks each dataframe in the input list for the presence of a 'Mass' or 'm/z' column.
        If either of these columns is found, it takes appropriate action based on the ionization mode.
        If neither column exists, it raises a ValueError.

        Args:
            input_dfs (list of pandas DataFrames): A list of dataframes to check.
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
                if "Mass" in df.columns:
                    pass  # 'Mass' column already exists, no action needed
                elif "m/z" in df.columns:
                    if ionizationMode == "Esi+":
                        df["Mass"] = df["m/z"] - 1.0073
                    elif ionizationMode == "Esi-":
                        df["Mass"] = df["m/z"] + 1.0073
                else:
                    raise ValueError("Either Mass or m/z column must be in the input file. (Check spelling!)")

        return

    def check_retention_time_column(self, input_dfs):
        """
        Check for the existence of alternate spellings of 'Retention_Time' column in input dataframes and rename to "Retention_Time".

        Args:
            input_dfs (list of pandas DataFrames): A list of dataframes to check.
        Raises:
            ValueError: If 'Retention_Time' column is not present in the input dataframe.
        Returns:
            None
        """
        # Iterate through list of dfs
        for df in input_dfs:
            if df is not None:
                # Check to see if there is not the expected spelling of "Retention_Time" column
                # if 'Retention_Time' not in df.columns:
                # replace alternative capitalizations
                df.rename(
                    columns={
                        "Retention_time": "Retention_Time",
                        "RETENTION_TIME": "Retention_Time",
                        "Retention time": "Retention_Time",
                        "Retention Time": "Retention_Time",
                    },
                    inplace=True,
                )
                # replace rt/RT
                df.rename(
                    columns={"rt": "Retention_Time", "RT": "Retention_Time"},
                    inplace=True,
                )
                # replace "Ret. Time" (SCIEX data)
                df.rename(columns={"Ret._Time": "Retention_Time"}, inplace=True)
                if "Retention_Time" not in df.columns:
                    raise ValueError("Retention_Time column must be in the input file. (Check spelling!)")

        return

    def create_analysis_parameters_sheet(self):
        """
        Create a dataframe to store analysis parameters in, assign parameters, then save
        as the "Analysis Parameters" sheet in the self.data_map dictionary.

        Args:
            None
        Notes:
            In a future version, we would like to add a "Version" key to be printed.
        Returns:
            None
        """
        # create a dataframe to store analysis parameters
        columns = ["Parameter", "Value"]
        df_analysis_parameters = pd.DataFrame(columns=columns)

        # loop through keys in self.parameters and log them
        for key in self.parameters:
            logger.info("key: {}".format(key))
            label = self.parameters[key][0]
            value = self.parameters[key][1]
            df_analysis_parameters.loc[len(df_analysis_parameters)] = [label, value]

        # add the dataframe to the data_map with the sheet name of 'Analysis Parameters'
        self.data_map["Analysis Parameters"] = df_analysis_parameters

        return

    def create_run_sequence_sheets(self):
        """
        If there are run sequence files submitted, create a sheet for each mode of run sequence file.
        This is for the AMOS visualizations to be able to grab the sequence information for
        run seequence plots

        Args:
            None
        Notes:
            In a future version, we would like to directly pass this information to AMOS separately rather
            than storing it in the results file where it increases the complexity of the results
        Returns:
            None
        """

        if self.run_sequence_pos_df is not None:
            self.data_map["Run Sequence (pos)"] = self.run_sequence_pos_df
        if self.run_sequence_neg_df is not None:
            self.data_map["Run Sequence (neg)"] = self.run_sequence_neg_df

        return

    def set_status(self, status, create=False):
        """
        Accepts a string (e.g., "Processing", or "Step Completed") and, if create is TRUE,
        post status to the logger with the Job ID and timestamp.

        Args:
            status (string)
            create (Boolean)
        Notes:
            In a future version, we would like to display status on the UI
        Returns:
            None
        """
        posts = self.mongo.posts
        # Get time stamp
        time_stamp = datetime.utcnow()
        # Generate ID with status
        post_id = self.jobid + "_" + "status"
        # If create, post update
        if create:
            posts.update_one(
                {"_id": post_id},
                {
                    "$set": {
                        "_id": post_id,
                        "date": time_stamp,
                        "status": status,
                        "error_info": "",
                    }
                },
                upsert=True,
            )
        else:
            posts.update_one(
                {"_id": post_id},
                {"$set": {"_id": post_id, "status": status}},
                upsert=True,
            )

    def set_except_message(self, e):
        """
        Accepts a string with error/exception information and then
        posts string to the logger with the Job ID and timestamp.

        Args:
            e (string)
        Notes:
            We are working to provide more detailed exceptions
        Returns:
            None
        """
        posts = self.mongo.posts
        # Get timestamp
        time_stamp = datetime.utcnow()
        # Generate ID with status
        post_id = self.jobid + "_" + "status"
        # Post string update
        posts.update_one(
            {"_id": post_id},
            {"$set": {"_id": post_id, "date": time_stamp, "error_info": e}},
            upsert=True,
        )

    def get_step(self):
        """
        Called if "try/ nta_run.execute()" (lines 112-113) fails; returns current
        step of the execute() function for diagnosing error.

        Args:
            None
        Returns:
            self.step (string)
        """
        # Return step string
        return self.step

    def assign_id(self):
        """
        Accesses self.dfs (list of dataframes) and calls task_fun.assign_feature_id()
        on each dataframe. Inserts column "Feature_ID" in the front of each dataframe
        with a unique numeric identifier for each row.

        Args:
            None
        Returns:
            None
        """
        # If both df exists, submit both to assign_feature_id() function, update start
        # kwarg by length of first df
        if self.dfs[0] is not None and self.dfs[1] is not None:
            self.dfs[0] = task_fun.assign_feature_id(self.dfs[0])
            self.dfs[1] = task_fun.assign_feature_id(self.dfs[1], start=len(self.dfs[0].index) + 1)
        # Else, submit single df to assign_feature_id() function
        elif self.dfs[0] is not None:
            self.dfs[0] = task_fun.assign_feature_id(self.dfs[0])
        else:
            self.dfs[1] = task_fun.assign_feature_id(self.dfs[1])
        return

    def pass_through_cols(self):
        """
        Accesses self.dfs (list of dataframes) and calls task_fun.passthrucol()
        on each dataframe. Identifies columns in dataframes that aren't needed for
        downstream functions; removes them from self.dfs and stores them in class
        object self.pass_through. These columns will get appended to the results.

        Args:
            None
        Notes:
            There is probably a more elegant solution than effectively running the
            task_fun.passthrucol() function twice, but I would run into errors if only
            one dataframe in self.dfs was present.
        Returns:
            None
        """
        # Submit dfs and all_headers to passthrucol() function, store outputs separately
        self.pass_through, self.dfs = zip(
            *[task_fun.passthrucol(df, self.all_headers) if df is not None else (None, None) for df in self.dfs]
        )
        return

    def filter_void_volume(self, min_rt):
        """
        Accesses self.dfs (list of dataframes) and self.parameters["minimum_rt"][1]
        then removes all rows with a value below "minimum_rt" in the "Retention_Time"
        column.

        Args:
            min_rt (float, user-submitted value with default of 0.0)
        Returns:
            None
        """
        # Iterate through dfs, removing rows where "Retention_Time" is below min_rt threshold
        self.dfs = [df.loc[df["Retention_Time"] > min_rt].copy() if df is not None else None for df in self.dfs]
        return

    def filter_duplicates(self):
        """
        Accesses self.dfs (list of dataframes), self.parameters["mass_accuracy_units"][1],
        self.parameters["mass_accuracy"][1], and self.parameters["rt_accuracy"][1].
        The task_fun.duplicates() function is called on each dataframe, with mass_accuracy,
        rt_accuracy, and ppm kwargs passed. The function identifies duplicate features
        within the passed tolerances, flagged in a new "Duplicates Features?" column.

        Args:
            None
        Returns:
            None
        """
        # Get ppm, mass_accuracy, and rt_accuracy parameters
        ppm = self.parameters["mass_accuracy_units"][1] == "ppm"
        mass_accuracy = float(self.parameters["mass_accuracy"][1])
        rt_accuracy = float(self.parameters["rt_accuracy"][1])
        # Perform duplicate flagging functions
        self.dfs = [
            task_fun.duplicates(df, mass_accuracy, rt_accuracy, ppm, self.blank_headers, self.sample_headers)
            if df is not None
            else None
            for df in self.dfs
        ]
        return

    def calc_statistics(self):
        """
        Accesses self.dfs (list of dataframes), self.parameters["mass_accuracy_units"][1],
        self.parameters["mass_accuracy"][1], self.parameters["rt_accuracy"][1], and self.parameters["mrl_std_multiplier"][1].
        The task_fun.chunk_stats() function is called on each dataframe, with mrl_std_multiplier passed. The function calculates
        mean, median, std_dev, CV, and replication % for each chemical feature across sample replicates. If adducts are
        selected by the user, the task_fun.adduct_identifier() function is called on each dataframe. Finally, the
        task_fun.column_sort_DFS() function is called to prepare the dataframes for output in the excel file.

        Args:
            None
        Returns:
            None
        """
        # Get ppm, mass_accuracy, rt_accuracy, and mrl_multiplier parameters
        ppm = self.parameters["mass_accuracy_units"][1] == "ppm"
        mass_accuracy = float(self.parameters["mass_accuracy"][1])
        rt_accuracy = float(self.parameters["rt_accuracy"][1])
        mrl_multiplier = float(self.parameters["mrl_std_multiplier"][1])
        # NTAW-509 Get miminum blank detection percentage
        min_blank_detection_percentage = float(self.parameters["min_replicate_hits_blanks"][1])
        # Iterate through dfs, calling chunk_stats() function
        # NTAW-49: Raises custom ValueError if blank columns are improperly named in the input dataframes
        try:
            self.dfs = [
                task_fun.chunk_stats(
                    df,
                    min_blank_detection_percentage,
                    self.blank_headers,
                    self.sample_headers,
                    mrl_multiplier=mrl_multiplier,
                )
                if df is not None
                else None
                for df in self.dfs
            ]
        except IndexError:
            raise ValueError(
                "Blank samples not found. Blanks must have one of the following text strings present: ['mb', 'Mb','mB', 'MB', 'blank', 'Blank', 'BLANK']"
            )
        # Get positive adducts, print to logger
        pos_adducts_selected = self.parameters["pos_adducts"][1]
        logger.info("pos adducts list: {}".format(self.parameters["pos_adducts"]))
        # Get negative adducts, print to logger
        neg_adducts_selected = self.parameters["neg_adducts"][1]
        logger.info("neg adducts list: {}".format(self.parameters["neg_adducts"]))
        # Get neutral losses, print to logger
        neutral_losses_selected = self.parameters["neutral_losses"][1]
        logger.info("neutral losses list: {}".format(self.parameters["neutral_losses"]))
        # Package adduct lists together into list of lists
        adduct_selections = [pos_adducts_selected, neg_adducts_selected, neutral_losses_selected]
        # Check if any adducts are selected
        to_run = False
        for li in adduct_selections:
            if len(li) > 0:
                to_run = True
        # Check if any adducts selected; if so, perform adduct identification functions
        if to_run:
            if self.dfs[0] is not None and self.dfs[1] is not None:
                self.dfs[0] = task_fun.adduct_identifier(
                    self.dfs[0], adduct_selections, mass_accuracy, rt_accuracy, ppm, ionization="positive"
                )
                self.dfs[1] = task_fun.adduct_identifier(
                    self.dfs[1], adduct_selections, mass_accuracy, rt_accuracy, ppm, ionization="negative"
                )
            elif self.dfs[0] is not None:
                self.dfs[0] = task_fun.adduct_identifier(
                    self.dfs[0], adduct_selections, mass_accuracy, rt_accuracy, ppm, ionization="positive"
                )
            else:
                self.dfs[1] = task_fun.adduct_identifier(
                    self.dfs[1], adduct_selections, mass_accuracy, rt_accuracy, ppm, ionization="negative"
                )
        # sort dataframe columns for data_feature_statistics (DFS) output
        if self.dfs[0] is not None and self.dfs[1] is not None:
            self.data_map["All Detection Statistics (Pos)"] = task_fun.column_sort_DFS(
                self.dfs[0], self.pass_through[0], self.all_headers
            )
            self.data_map["All Detection Statistics (Neg)"] = task_fun.column_sort_DFS(
                self.dfs[1], self.pass_through[1], self.all_headers
            )
        elif self.dfs[0] is not None:
            self.data_map["All Detection Statistics (Pos)"] = task_fun.column_sort_DFS(
                self.dfs[0], self.pass_through[0], self.all_headers
            )
        else:
            self.data_map["All Detection Statistics (Neg)"] = task_fun.column_sort_DFS(
                self.dfs[1], self.pass_through[1], self.all_headers
            )
        return

    def store_scatterplots(self):
        # Store in class variable
        self.cv_scatterplots_out.append(
            cv_scatterplot(
                parameters=self.parameters,
                data_map=self.data_map,
                blank_headers=self.blank_headers,
                sample_headers=self.sample_headers,
            )
        )
        # Map to outputs
        self.cv_scatterplot_map["cv_scatterplot"] = self.cv_scatterplots_out[0]
        project_name = self.parameters["project_name"][1]
        self.gridfs.put(
            "&&".join(self.cv_scatterplot_map.keys()),
            _id=self.jobid + "_cv_scatterplots",
            encoding="utf-8",
            project_name=project_name,
        )
        # Save to MongoDB
        self.mongo_save(self.cv_scatterplot_map["cv_scatterplot"], step="cv_scatterplot")

    def store_heatmap(self):
        # Store in class variable
        self.occurrence_heatmaps_out.append(
            occurrence_heatmap(
                parameters=self.parameters,
                data_map=self.data_map,
                blank_headers=self.blank_headers,
                sample_headers=self.sample_headers,
            )
        )
        # Map to outputs
        self.occurrence_heatmap_map["occurrence_heatmap"] = self.occurrence_heatmaps_out[0]
        project_name = self.parameters["project_name"][1]
        self.gridfs.put(
            "&&".join(self.occurrence_heatmap_map.keys()),
            _id=self.jobid + "_occurrence_heatmaps",
            encoding="utf-8",
            project_name=project_name,
        )
        # Save to MongoDB
        self.mongo_save(self.occurrence_heatmap_map["occurrence_heatmap"], step="occurrence_heatmap")

    def check_tracers(self):
        """
        Check for tracers. If present, access user input parameters for mass accuracy units,
        mass accuracy, y-axis scaling (run sequence plots), and trendline toggling (yes/no).
        Access self.dfs and pass each dataframe to task_fun.check_feature_tracers() with
        other parameters as kwargs. Call fromat_tracer_file() function imported from
        utilities.py, then instantiate run sequence plots with the WebApp_plotter. Access
        pass-through columns (if present), and call task_fun.column_sort_TSR() function
        to format the tracer outputs for the excel file.

        Args:
            None
        Returns:
            None
        """
        # If no tracer file present, skip step
        if self.tracer_df is None:
            logger.info("No tracer file, skipping this step.")
            return
        if self.verbose:
            logger.info("Tracer file found, checking tracers.")
        # Get ppm, mass_accuracy_tr, yaxis_scale, and trendline_shown parameters
        ppm = self.parameters["mass_accuracy_units_tr"][1] == "ppm"
        mass_accuracy_tr = float(self.parameters["mass_accuracy_tr"][1])
        rt_accuracy_tr = float(self.parameters["rt_accuracy_tr"][1])
        yaxis_scale = self.parameters["tracer_plot_yaxis_format"][1]
        trendline_shown = self.parameters["tracer_plot_trendline"][1] == "yes"
        # Perform check_tracers() on dfs with tracer file
        self.tracer_dfs_out, self.dfs = zip(
            *[
                task_fun.check_feature_tracers(
                    df, self.tracer_df, mass_accuracy_tr, rt_accuracy_tr, ppm, self.blank_headers, self.sample_headers
                )
                if df is not None
                else (None, None)
                for df in self.dfs
            ]
        )
        # Call format_tracer_file imported from utilities.py
        self.tracer_dfs_out = [
            task_fun.format_tracer_file(df) if df is not None else None for df in self.tracer_dfs_out
        ]

        # Declare plotter
        df_WA = WebApp_plotter()

        # Create plot
        if self.tracer_dfs_out[0] is not None:
            # Troubleshooting NTAW-460
            logger.info("self.tracer_dfs_out[1] shape= {}".format(self.tracer_dfs_out[0].shape))
            # logger.info("self.tracer_dfs_out[1] columns= {}".format(self.tracer_dfs_out[0].columns.values))

            # Check that run sequence file is present. Raise IndexError if no run sequence file is found.
            if self.run_sequence_pos_df is None:
                raise IndexError(
                    "A run sequence file was not found. Please input a corresponding run sequence file for each input data file."
                )
            # If run sequence file present, check sample names in run sequence against those in the dataframe.
            else:
                # Pass positive mode data and positive run sequence file to check_run_seq()
                task_fun.check_run_seq(self.dfs[0], self.run_sequence_pos_df, self.blank_headers, self.sample_headers)

            listOfPNGs, df_debug, debug_list = df_WA.make_seq_scatter(
                df_in=self.tracer_dfs_out[0],
                df_seq=self.run_sequence_pos_df,
                ionization="pos",
                y_scale=yaxis_scale,
                fit=trendline_shown,
                share_y=False,
                # y_fixed=False,
                y_step=6,
                same_frame=False,
                legend=True,
                # chemical_names=None,
                dark_mode=False,
            )
            self.tracer_plots_out.append(listOfPNGs)

        else:
            self.tracer_plots_out.append(None)

        # Declare plotter
        df_WA = WebApp_plotter()

        # Create plot
        if self.tracer_dfs_out[1] is not None:
            logger.info("self.tracer_dfs_out[1] shape= {}".format(self.tracer_dfs_out[1].shape))
            # logger.info("self.tracer_dfs_out[1] columns= {}".format(self.tracer_dfs_out[1].columns.values))

            # Check that run sequence file is present. Raise IndexError if no run sequence file is found.
            if self.run_sequence_neg_df is None:
                raise IndexError(
                    "A run sequence file was not found. Please input a corresponding run sequence file for each input data file."
                )
            # If run sequence file present, check sample names in run sequence against those in the dataframe.
            else:
                # Pass negative mode data and negative run sequence file to check_run_seq()
                task_fun.check_run_seq(self.dfs[1], self.run_sequence_neg_df, self.blank_headers, self.sample_headers)

            listOfPNGs, df_debug, debug_list = df_WA.make_seq_scatter(
                df_in=self.tracer_dfs_out[1],
                df_seq=self.run_sequence_neg_df,
                ionization="neg",
                y_scale=yaxis_scale,
                fit=trendline_shown,
                share_y=False,
                # y_fixed=False,
                y_step=6,
                same_frame=False,
                legend=True,
                # chemical_names=None,
                dark_mode=False,
            )

            self.tracer_plots_out.append(listOfPNGs)
            logger.info("df_debug shape= {}".format(df_debug.shape))
            # logger.info("df_debug columns= {}".format(df_debug.columns.values))

        else:
            self.tracer_plots_out.append(None)

        # Combine items of tracer_dfs_out list
        dft = pd.concat([self.tracer_dfs_out[0], self.tracer_dfs_out[1]])
        # If passthrough columns present, combine
        if self.pass_through[0] is not None and self.pass_through[1] is not None:
            passthru = pd.concat([self.pass_through[0], self.pass_through[1]])
        elif self.pass_through[0] is not None:
            passthru = self.pass_through[0]
        else:
            passthru = self.pass_through[1]
        # Update logger
        # logger.info("tracer dft= {}".format(dft.columns.tolist()))
        # logger.info("tracer passthru= {}".format(passthru.columns.tolist()))
        # Combine and sort processed tracer file and combined passthrough columns
        self.data_map["Tracer Detection Statistics"] = task_fun.column_sort_TSR(dft, passthru)

        if self.tracer_plots_out[0] is not None:
            for i in range(len(self.tracer_plots_out[0])):
                self.tracer_map["tracer_plot_pos_" + str(i + 1)] = self.tracer_plots_out[0][i]

        # Add an if statement below to account for: if only negative mode data is entered, and
        # only a negative tracer file is submitted, tracer_plots_out will only have one entry at [0]
        if len(self.tracer_plots_out) > 1:
            if self.tracer_plots_out[1] is not None:
                for i in range(len(self.tracer_plots_out[1])):
                    self.tracer_map["tracer_plot_neg_" + str(i + 1)] = self.tracer_plots_out[1][i]

        # Convert the figure objects in tracer_map into PNGs that can be stored in gridfs
        for key in self.tracer_map.keys():
            buf = io.BytesIO()
            # Save the figure in buffer as png
            self.tracer_map[key].savefig(buf, bbox_inches="tight", format="png")
            buf.seek(0)
            self.tracer_map[key] = buf.read()
        # Set project name
        project_name = self.parameters["project_name"][1]
        self.gridfs.put(
            "&&".join(self.tracer_map.keys()),
            _id=self.jobid + "_tracer_files",
            encoding="utf-8",
            project_name=project_name,
        )
        for key in self.tracer_map.keys():
            self.mongo_save(self.tracer_map[key], step=key)
        return

    def perform_qNTA(self):
        """
        Call task_functions qNTA_preprocessing() to transform qNTA_Surrogate_Input_File
        into qNTA_Surrogate_Detection_Statistics_File.

        Args:
            self
        Returns:
            None
        """
        # Define controls
        controls = [
            float(self.parameters["min_replicate_hits"][1]),
            float(self.parameters["max_replicate_cv"][1]),
            float(self.parameters["min_replicate_hits_blanks"][1]),
            float(self.parameters["mrl_std_multiplier"][1]),
        ]
        # Call task_fun.qnta_preprocessing(), get sds
        do_qNTA = self.parameters["do_qnta"][1] == "yes"
        # Get run parameters
        ppm = self.parameters["mass_accuracy_units_tr"][1] == "ppm"
        mass_accuracy_tr = float(self.parameters["mass_accuracy_tr"][1])
        rt_accuracy_tr = float(self.parameters["rt_accuracy_tr"][1])
        # Perform check_tracers() on dfs with tracer file
        self.qnta_dfs_out, self.dfs, self.qnta_occ_input = zip(
            *[
                task_fun.qnta_preprocessing(
                    df,
                    self.qnta_df,
                    passthru,
                    mass_accuracy_tr,
                    rt_accuracy_tr,
                    ppm,
                    controls,
                    self.blank_headers,
                    self.sample_headers,
                )
                if df is not None
                else (None, None, None)
                for df, passthru in zip(self.dfs, self.pass_through)
            ]
        )
        # Combine Surrogate Detection Statistics from separate modes
        # Combine items of tracer_dfs_out list
        if self.qnta_dfs_out[0] is not None and self.qnta_dfs_out[1] is not None:
            sds = pd.concat([self.qnta_dfs_out[0], self.qnta_dfs_out[1]])
        elif self.qnta_dfs_out[0] is not None:
            sds = self.qnta_dfs_out[0].copy()
        else:
            sds = self.qnta_dfs_out[1].copy()
        # Add Surrogate Detection Statistics to the qnta datamap
        self.qnta_map["Surrogate Detection Statistics"] = sds
        # Instatiate qNTA objects below
        # If both modes, instantiate both objects and execute both objects
        if self.qnta_dfs_out[0] is not None and self.qnta_dfs_out[1] is not None:
            # Check if self.val_df is not None
            if self.val_df is not None:
                # Separate into mode data
                val_pos = self.val_df.loc[self.val_df["Ionization_Mode"] == "ESI+", :]
                val_neg = self.val_df.loc[self.val_df["Ionization_Mode"] == "ESI-", :]
            else:
                # Assign Nones
                val_pos = None
                val_neg = None
            # Create qNTAClass object for pos mode data
            qnta_pos = qNTAClass(
                self.qnta_dfs_out[0],
                validation_input=val_pos,
                occurrence_input=self.qnta_occ_input[0],
                parameters=None,
            )
            # Execute class functions
            qnta_pos.execute()
            # Create qNTAClass object for neg mode data
            qnta_neg = qNTAClass(
                self.qnta_dfs_out[1],
                validation_input=val_neg,
                occurrence_input=self.qnta_occ_input[1],
                parameters=None,
            )
            # Execute class functions
            qnta_neg.execute()
            # Combine cc_metrics outputs, store in qnta datamap
            self.qnta_map["Calibration Curve Metrics"] = pd.concat([qnta_pos.cc_metrics, qnta_neg.cc_metrics])
            # Get RF percentiles, store in qnta datamap
            self.qnta_map["Pos Mode RF Percentile Ests"] = qnta_pos.RF_percs
            self.qnta_map["Neg Mode RF Percentile Ests"] = qnta_neg.RF_percs
            # Check status of validation_out
            if qnta_pos.validation_out is not None:
                # Get validation outputs, store in qnta datamap
                self.qnta_map["Pos Validation Output"] = task_fun.validation_col_rename(qnta_pos.validation_out)
                # Generate AQ plots
                self.store_aq_plots(qnta_object=qnta_pos, im="ESI+")
                # Check status of summary_out
                if qnta_pos.summary_out is not None:
                    # Get validation outputs, store in qnta datamap
                    self.qnta_map["Pos Validation Summary"] = qnta_pos.summary_out
            # Check status of validation_out
            if qnta_neg.validation_out is not None:
                # Get validation outputs, store in qnta datamap
                self.qnta_map["Neg Validation Output"] = task_fun.validation_col_rename(qnta_neg.validation_out)
                # Generate AQ plots
                self.store_aq_plots(qnta_object=qnta_neg, im="ESI-")
                # Check status of summary_out
                if qnta_neg.summary_out is not None:
                    # Get validation outputs, store in qnta datamap
                    self.qnta_map["Neg Validation Summary"] = qnta_neg.summary_out

        # If only positive mode, instatiate positive mode and execute object
        elif self.qnta_dfs_out[0] is not None:
            # Check if self.val_df is not None
            if self.val_df is not None:
                # Separate into mode data
                val_pos = self.val_df.loc[self.val_df["Ionization_Mode"] == "ESI+", :]
            else:
                # Assign None
                val_pos = None
            # Create qNTAClass object for pos mode data
            qnta_pos = qNTAClass(
                self.qnta_dfs_out[0],
                validation_input=val_pos,
                occurrence_input=self.qnta_occ_input[0],
                parameters=None,
            )
            # Execute class functions
            qnta_pos.execute()
            # Get cc_metrics output, store in qnta datamap
            self.qnta_map["Calibration Curve Metrics"] = qnta_pos.cc_metrics
            # Get RF percentiles, store in qnta datamap
            self.qnta_map["Pos Mode RF Percentile Ests"] = qnta_pos.RF_percs
            # Check status of validation_out
            if qnta_pos.validation_out is not None:
                # Get validation outputs, store in qnta datamap
                self.qnta_map["Pos Validation Output"] = task_fun.validation_col_rename(qnta_pos.validation_out)
                # Generate AQ plots
                self.store_aq_plots(qnta_object=qnta_pos, im="ESI+")
                # Check status of summary_out
                if qnta_pos.summary_out is not None:
                    # Get validation outputs, store in qnta datamap
                    self.qnta_map["Pos Validation Summary"] = qnta_pos.summary_out

        # If only negative mode, instantiate negative mode and execute object
        else:
            # Check if self.val_df is not None
            if self.val_df is not None:
                # Separate into mode data
                val_neg = self.val_df.loc[self.val_df["Ionization_Mode"] == "ESI-", :]
            else:
                # Assign None
                val_neg = None
            # Create qNTAClass object for neg mode data
            qnta_neg = qNTAClass(
                self.qnta_dfs_out[1],
                validation_input=val_neg,
                occurrence_input=self.qnta_occ_input[1],
                parameters=None,
            )
            # Execute class functions
            qnta_neg.execute()
            # Get cc_metrics output, store in qnta datamap
            self.qnta_map["Calibration Curve Metrics"] = qnta_neg.cc_metrics
            # Get RF percentiles, store in qnta datamap
            self.qnta_map["Neg Mode RF Percentile Ests"] = qnta_neg.RF_percs
            # Check status of validation_out
            if qnta_neg.validation_out is not None:
                # Get validation outputs, store in qnta datamap
                self.qnta_map["Neg Validation Output"] = task_fun.validation_col_rename(qnta_neg.validation_out)
                # Generate AQ plots
                self.store_aq_plots(qnta_object=qnta_neg, im="ESI-")
                # Check status of summary_out
                if qnta_neg.summary_out is not None:
                    # Get validation outputs, store in qnta datamap
                    self.qnta_map["Neg Validation Summary"] = qnta_neg.summary_out

    def store_aq_plots(
        self,
        qnta_object,
        im,
    ):
        # Store in class variable
        self.aq_plots_out.append(
            qnta_object.AQ_plots(
                validation_out=qnta_object.validation_out,
                long_form=qnta_object.parameters["long_form"],
                LOO=qnta_object.parameters["LOO"],
            )
        )
        # Define plot string
        name = "AQ_plots_" + im
        # Map to outputs
        self.aq_plots_map[name] = self.aq_plots_out[0]
        project_name = self.parameters["project_name"][1]
        self.gridfs.put(
            "&&".join(self.aq_plots_map.keys()),
            _id=self.jobid + name,
            encoding="utf-8",
            project_name=project_name,
        )
        # Save to MongoDB
        self.mongo_save(self.aq_plots_map[name], step=name)
        # Reset self.aq_plots_out and self.aq_plots_map
        self.aq_plots_out = []
        self.aq_plots_map = {}

    def clean_features(self):
        """
        Accesses self.dfs and the user-input parameters for min replicate hits, max CV,
        and mrl std_dev multiplier. Check for presence of tracer file, then call task_fun.clean_features()
        that uses the parameters to document filtering and flagging the the Decision
        Documentation excel sheet. Call task_fun.Blank_Subtract_Mean() to subtract each feature's
        blank value from the mean of each sample grouping.

        Args:
            None
        Returns:
            None
        """
        # Define controls
        controls = [
            float(self.parameters["min_replicate_hits"][1]),
            float(self.parameters["max_replicate_cv"][1]),
            float(self.parameters["min_replicate_hits_blanks"][1]),
            float(self.parameters["mrl_std_multiplier"][1]),
        ]
        # Pass inputs to clean_features() and store docs, dfs_flagged, and finally dfs
        self.dfs, self.docs, self.dfs_flagged = zip(
            *[task_fun.clean_features(df, controls) if df is not None else (None, None, None) for df in self.dfs]
        )
        # Check if CVs are filtered or flagged, manipulate corresponding dfs
        if self.parameters["filter_cv"][1] == "yes":
            # if "filter_cv" not in self.parameters or self.parameters["filter_cv"][1] == "yes":
            # subtract blanks from means
            self.dfs = [task_fun.Blank_Subtract_Mean(df) if df is not None else None for df in self.dfs]
        else:
            # subtract blanks from means
            self.dfs_flagged = [task_fun.Blank_Subtract_Mean(df) if df is not None else None for df in self.dfs_flagged]
        return

    def merge_columns_onto_tracers(self):
        """
        Check for tracer files. If present, subset tracer dataframe with specific
        order of columns desired for the Tracer Summary output.

        Args:
            None
        Returns:
            None
        """
        # Check for tracer file, if not present return
        if self.tracer_df is None:
            logger.info("No tracer file, skipping this step.")
            return
        # Grab the tracer dataframe from self.data_map
        dft = self.data_map["Tracer Detection Statistics"]
        # logger.info("dft: {}".format(dft.columns))
        # create summary table
        if "DTXSID" not in dft.columns:
            dft["DTXSID"] = ""
        # Subset dataframe with columns of interest
        dft = dft[
            [
                "Feature ID",
                "Chemical Name",
                "DTXSID",
                "Ionization Mode",
                "Mass Error (PPM)",
                "Retention Time Difference",
                "Total Detection Count",
                "Total Detection Percentage",
                "Max CV Across Samples",
            ]
        ]
        # Map dataframe to Tracer Summary output
        self.data_map["Tracer Summary"] = dft
        return

    def combine_modes(self):
        """
        For self.dfs and self.dfs_flagged, call task_fun.combine(). Checks for presence
        of pos and neg data modes, and if both present combine them. For self.doc_combined,
        call task_fun.combine_doc() function to check presence of both data modes and combine
        if present. Finally, merge passthrough columns back on to processed data (df_combined)
        and decision documentation sheet (doc_combined) by calling task_fun.MPP_ready().
        Store final dataframes in self.data_map dictionary.

        Args:
            None
        Returns:
            None
        """
        # Check for tracer file
        tracer_df_bool = False
        if self.tracer_df is not None:
            tracer_df_bool = True
        # Check if CVs are filtered or flagged, manipulate corresponding dfs
        if self.parameters["filter_cv"][1] == "yes":
            # combine dfs from both modes
            self.df_combined = task_fun.combine(self.dfs[0], self.dfs[1])
        else:
            # combine df_flaggeds from both modes
            self.df_flagged_combined = task_fun.combine(self.dfs_flagged[0], self.dfs_flagged[1])
        # combine docs from both modes
        self.doc_combined = task_fun.combine_doc(self.docs[0], self.docs[1], tracer_df=tracer_df_bool)
        # Replace zero values of "Selected MRL" column with blank cells prior to exporting to data_map
        self.doc_combined["Selected MRL"] = self.doc_combined["Selected MRL"].replace(0, "")
        # Also replace zero values of blank mean columns with blank cells prior to exporting to data_map
        blanks = ["MB", "mb", "mB", "Mb", "blank", "Blank", "BLANK"]
        Mean = self.doc_combined.columns[self.doc_combined.columns.str.contains(pat="Mean ")].tolist()
        Mean_MB = [md for md in Mean if any(x in md for x in blanks)]
        self.doc_combined[Mean_MB] = self.doc_combined[Mean_MB].replace(0, "")
        # Map to Decision Documentation output
        self.data_map["Decision Documentation"] = self.doc_combined
        # Check if CVs are filtered or flagged, manipulate corresponding dfs
        if self.parameters["filter_cv"][1] == "yes":
            # Prep combined df for output by combining with passthrough cols and formatting
            self.mpp_ready = task_fun.MPP_Ready(
                self.df_combined,
                self.pass_through,
                self.blank_headers,
                self.sample_headers,
            )
            # Map df outputs to Final Occurrence Matrices sheets
            self.data_map["Final Occurrence Matrix"] = reduced_file(
                self.mpp_ready, self.blank_headers, self.sample_headers
            )
        else:
            # Prep combined df_flagged for output by combining with passthrough cols and formatting
            self.mpp_ready_flagged = task_fun.MPP_Ready(
                self.df_flagged_combined,
                self.pass_through,
                self.blank_headers,
                self.sample_headers,
            )
            # Map df_flagged outputs to Final Occurrence Matrices sheets
            self.data_map["Final Occurrence Matrix"] = reduced_file(
                self.mpp_ready_flagged, self.blank_headers, self.sample_headers
            )

    def perform_dashboard_search(self, lower_index=0, upper_index=None, save=True):
        """
        Access self.df_flagged_combined to get full list of chemical features that passed
        replication and MRL thresholds (flagged_combined keeps CV fails). Subset the dataframe
        by features flagged "For_Dashboard_Search". Check user-selected "search mode" - if "mass",
        call task_fun.masses() function to format masses for search. If "formulas", call
        task_fun.formulas() funcion to format formulas for search. Send formatted list to
        api_search_masses_batch() function imported from utilities.py. Handle returned JSON
        and combine back into results. Finally, call task_fun.calc_toxcast_percent_active()
        function to get tox information, and combine with dashboard results in "Chemical Results" output.

        Args:
            lower_index (integer; start search range at this Feature_ID)
            upper_index (interger, default=None; end search range at this Feature_ID)
        Returns:
            None
        """
        # Check if CVs are filtered or flagged, manipulate corresponding dfs
        if self.parameters["filter_cv"][1] == "yes":
            # Get subset of features to search from df_flagged_combined
            to_search = self.df_combined.loc[self.df_combined["For_Dashboard_Search"] == "1", :].copy()
        else:
            # Get subset of features to search from df_flagged_combined
            to_search = self.df_flagged_combined.loc[self.df_flagged_combined["For_Dashboard_Search"] == "1", :].copy()
        # Check if searching by mass or formula, drop duplicates
        if self.parameters["search_mode"][1] == "mass":
            to_search = to_search.drop_duplicates(
                subset="Mass",
                keep="first",
            )
        else:
            to_search = to_search.drop_duplicates(
                subset="Formula",
                keep="first",
            )
        # Calculate number of fragments to search
        n_search = len(to_search)
        # Update logger
        logger.info("Total # of queries: {}".format(n_search))
        # Subset features to search
        to_search = to_search.iloc[lower_index:upper_index, :]
        # If searching by mass, call api_search_masses_batch()
        if self.parameters["search_mode"][1] == "mass":
            mono_masses = task_fun.masses(to_search)
            dsstox_search_df = api_search_masses_batch(
                mono_masses,
                float(self.parameters["parent_ion_mass_accuracy"][1]),
                batchsize=150,
                jobid=self.jobid,
            )
            logger.info("===========DSSTox retrieval returned to nta_task===========")
        # If searching by formula, call api_search_masses_batch()
        else:
            formulas = task_fun.formulas(to_search)
            response = api_search_formulas(formulas, self.jobid)
            if not response.ok:  # check if we got a successful response
                raise requests.exceptions.HTTPError(
                    "Unable to access DSSTOX API. Please contact an administrator or try turning the DSSTox search option off."
                )
            dsstox_search_json = io.StringIO(json.dumps(response.json()["results"]))
            dsstox_search_df = pd.read_json(
                dsstox_search_json,
                orient="split",
                dtype={"TOXCAST_NUMBER_OF_ASSAYS/TOTAL": "object"},
            )
        # Check if CVs are filtered or flagged, manipulate corresponding dfs
        if self.parameters["filter_cv"][1] == "yes":
            # Merge API search results with mpp_ready_flagged dataframe ID, mass, and RT
            dsstox_search_df = self.mpp_ready[["Feature ID", "Mass", "Retention Time"]].merge(
                dsstox_search_df, how="right", left_on="Mass", right_on="INPUT"
            )
        else:
            # Merge API search results with mpp_ready_flagged dataframe ID, mass, and RT
            dsstox_search_df = self.mpp_ready_flagged[["Feature ID", "Mass", "Retention Time"]].merge(
                dsstox_search_df, how="right", left_on="Mass", right_on="INPUT"
            )
        logger.info("===========API results merged with mpp_ready dataframe===========")
        # Calculate toxcast_percent_active values
        dsstox_search_df = task_fun.calc_toxcast_percent_active(dsstox_search_df)
        logger.info("===========Toxcast % active calculated===========")
        # Perform atom filtering on the DSSTox results
        if self.parameters["atom_ranges"][1]:
            dsstox_search_df = task_fun.DSSTox_atom_filtering(dsstox_search_df, self.parameters["atom_ranges"][1])
            logger.info("===========Atom filtering completed===========")
        # Perform data source filtering on the DSSTox results
        if self.parameters["return_nods"][1] == "no":
            dsstox_search_df = dsstox_search_df.loc[dsstox_search_df["SOURCE_COUNT"] > 1, :]
            logger.info("===========Source Count filtering completed===========")
        # Map dataframe to Chemical Results output
        self.data_map["Chemical Results"] = dsstox_search_df
        logger.info("===========API Results mapped to self.data_map===========")
        # Store search results
        self.search_results = dsstox_search_df
        logger.info("===========API Results stored in self.search_results===========")

    def perform_hcd_search(self):
        """
        Check length of Dashboard Search Results. If length > 0, format Dashboard Search
        Results into list of unique DTXSIDs and call batch_search_hcd() function imported
        from utilities.py to retrieve hazard data from the Hazard Comparison Dashboard.
        Merge HCD results with Dashboard results and update "Chemical Results" output.

        Args:
            None
        Returns:
            None
        """
        # Update logger
        logger.info(f"Querying the HCD with DTXSID identifiers")
        # Check length of search results from API batch search
        if len(self.search_results) > 0:
            # Get unique DTXSIDs from searched features
            dtxsid_list = self.search_results["DTXSID"].unique()
            # Hit HCD API with unique DTXSIDs, retrieve hazard info
            hcd_results = batch_search_hcd(dtxsid_list)
            # Merge retrieved hazard info with dashboard search results
            self.search_results = self.search_results.merge(hcd_results, how="left", on="DTXSID")
            # Update Chemical Results output
            self.data_map["Chemical Results"] = self.search_results

    def add_hazard_cols(self, df):
        # Define authority and hazard score column names
        authority_cols = df.filter(like="authority").columns
        score_cols = df.filter(like="score").columns

        # Define mapping of authority and hazard score categorical values to numerical values
        authority_mapping = {"Authoritative": 3, "Screening": 2, "QSAR Model": 1}
        score_mapping = {"VH": 4, "H": 3, "M": 2, "L": 1, "I": np.nan, "ND": np.nan}

        # Create mapped column names
        authority_mapped_cols = [f"{col}_mapped" for col in authority_cols]
        score_mapped_cols = [f"{col}_mapped" for col in score_cols]

        # Apply mappings and store in mapped columns
        df[authority_mapped_cols] = df[authority_cols].applymap(lambda x: authority_mapping.get(x, x))
        df[score_mapped_cols] = df[score_cols].applymap(lambda x: score_mapping.get(x, x))

        # Set authority mapped column to NaN where corresponding hazard mapped score column is NaN for all hazard endpoints
        for score_mapped_col in score_mapped_cols:
            authority_mapped_col = score_mapped_col.replace("score", "authority")
            df.loc[df[score_mapped_col].isna(), authority_mapped_col] = np.nan

        # Calculate the Quality Adjusted Hazard Score for each substance (Mean authority score* Mean hazard score)
        df["pre_Hazard Score"] = df[authority_mapped_cols].mean(axis=1) * df[score_mapped_cols].mean(axis=1)

        # Calculate the Completeness Score for each substance. (Number of endpoints that have data / total number of endpoints)
        endpoints = [col.split("_")[0] for col in df.columns if col.endswith("authority")]
        num_endpoints = len(endpoints)
        num_endpoints_with_data = df[[f"{endpoint}_score_mapped" for endpoint in endpoints]].notna().sum(axis=1)
        df["pre_completeness"] = num_endpoints_with_data / num_endpoints

        # Collapse on structure by keeping the highest QAH score of all substances associated with each DTXCID.
        df["Hazard Score"] = df.groupby(["Feature ID", "DTXCID_INDIVIDUAL_COMPONENT"])["pre_Hazard Score"].transform(
            "max"
        )

        # Keep the highest completeness score associated with each structure's QAH_COLLAPSED value
        def func(group):
            max_qah = group["pre_Hazard Score"].max()
            return group.loc[group["pre_Hazard Score"] == max_qah, "pre_completeness"].max()

        col_comp = df.groupby(["Feature ID", "DTXCID_INDIVIDUAL_COMPONENT"]).apply(func).reset_index()

        col_comp.rename(columns={0: "Hazard Completeness Score"}, inplace=True)
        df = df.merge(col_comp, on=["Feature ID", "DTXCID_INDIVIDUAL_COMPONENT"], how="left")

        # df = df.drop(authority_mapped_cols + score_mapped_cols + ["pre_Hazard Score", "pre_completeness"], axis=1)
        df = df.drop(["pre_Hazard Score", "pre_completeness"], axis=1)

        return df

    def mongo_save(self, file, step=""):
        """
        Take file chunk, and if it is a dataframe, arrange it via JSON to save in Mongo.
        Else, save separately.

        Args:
            file (any file type)
        Returns:
            None
        """
        # Check if file is type Pandas dataframe
        if isinstance(file, pd.DataFrame):
            # Convert to JSON
            to_save = file.to_json(orient="split")
        else:
            # Else handle with no conversion
            to_save = file
        # Format id
        id = self.jobid + "_" + step
        # Get project name
        project_name = self.parameters["project_name"][1]
        # Save item to MongoDB using item id
        self.gridfs.put(to_save, _id=id, encoding="utf-8", project_name=project_name)

    def save_excel_to_mongo(self):
        # Get project name
        project_name = self.parameters["project_name"][1]
        # Obtain a list of all keys in the data map (These will become the excel workbook sheet names)
        keys_list = list(self.data_map.keys())

        # save the csv for QAQC visuals to mongo
        self.save_QAQC_csv_to_mongo()

        # Check for 'Chemical Results' in keys list
        if "Chemical Results" in keys_list:
            # Replaces the static DTXSIDs in the DTXSID column with the corresponding hyperlinks.
            # self.data_map["Chemical Results"]["DTXSID"] = self.data_map["Chemical Results"]["DTXSID"].apply(
            #     lambda x: make_hyperlink(x)
            # )
            # Adds another column to hold the CompTox links.
            logger.info("===========Convert DTXSIDs to hyperlinks===========")
            self.data_map["Chemical Results"].insert(
                loc=8,
                column="CompTox links",
                value=self.data_map["Chemical Results"]["DTXSID"].apply(lambda x: make_hyperlink(x)),
            )

            # NTAW-755
            logger.info("===========Add collapsed and normalized metadata columns===========")

            # Add metadata columns to chemical results dataframe
            metadata_fields = ["PATENT", "LITERATURE", "PUBMED", "SOURCE"]
            i = 19
            for field in metadata_fields:
                self.data_map["Chemical Results"].insert(
                    i,
                    f"{field}_COUNT_COLLAPSED",
                    self.data_map["Chemical Results"]
                    .groupby(["Feature ID", "DTXCID_INDIVIDUAL_COMPONENT"])[f"{field}_COUNT"]
                    .transform("sum"),
                )
                self.data_map["Chemical Results"].insert(
                    i + 1,
                    f"{field}_COUNT_COLLAPSED_NORM",
                    self.data_map["Chemical Results"][f"{field}_COUNT_COLLAPSED"]
                    / self.data_map["Chemical Results"]
                    .groupby("Feature ID")[f"{field}_COUNT_COLLAPSED"]
                    .transform("max"),
                )
                i = i + 3
            # If hazard search was performed, add the relevant hazard calculation columns to the chemical results dataframe
            if self.parameters["search_hcd"][1] == "yes":
                logger.info("===========Add hazard columns===========")
                self.data_map["Chemical Results"] = self.add_hazard_cols(self.data_map["Chemical Results"])

            # Save the metadata/hazard csv to MongoDB
            logger.info("===========Saving tripod CSV to mongo===========")
            self.save_tripod_csv_to_mongo()

            # If hazard search was performed, remove the mapped hazard column from the dataframe,
            # since these columns have already been passed to the results csv in save_tripod_csv_to_mongo()
            if self.parameters["search_hcd"][1] == "yes":
                columns_to_drop = [col for col in self.data_map["Chemical Results"].columns if col.endswith("mapped")]
                self.data_map["Chemical Results"] = self.data_map["Chemical Results"].drop(columns=columns_to_drop)

            # Check length of "Chemical Results"
            sheet_limit = 500000
            # If "Chemical Results" is bigger than limit, chunk into smaller sizes
            if len(self.data_map["Chemical Results"]) > sheet_limit:
                # Set counter
                i = 1
                # Iterate through "Chemical Results", saving new chunks into 'chem_res_map'
                for chunk in task_fun.chunk_dataframe(self.data_map["Chemical Results"], sheet_limit):
                    # Assemble sheet name
                    sheet = "Chemical Results " + str(i)
                    # Assign dataframe chunk to sheet dict key
                    self.chem_res_map[sheet] = chunk
                    # Increase counter
                    i += 1
                logger.info("===========Chemical Results split into separate sheets===========")
            else:
                # If "Chemical Results" is under limit, save into 'chem_res_map'
                self.chem_res_map["Chemical Results"] = self.data_map["Chemical Results"]
            # Remove key from 'data_map'
            del self.data_map["Chemical Results"]

            # Create excel book from Chemical Results
            # Save project name to MongoDB using jobid
            self.gridfs.put(project_name, _id=f"{self.jobid}_project_name_chemical_results", encoding="utf-8")
            # Save results excel file to MongoDB using id
            id = self.jobid + "_excel_chem"
            self.gridfs.put(task_fun.create_excel_book(self.chem_res_map, chem_res=True), _id=id)
            self.log_memory_usage("create_excel_book from chem_res_map")

            logger.info("===========Saved Chemical Results excel book to MongoDB===========")
        # Create excel book for QAQC
        # Save project name to MongoDB using jobid
        self.gridfs.put(project_name, _id=f"{self.jobid}_project_name_QAQC", encoding="utf-8")
        # Save results excel file to MongoDB using id
        id = self.jobid + "_excel_qaqc"
        self.gridfs.put(task_fun.create_excel_book(self.data_map, chem_res=False), _id=id)
        self.log_memory_usage("create_excel_book from data_map")
        logger.info("===========Saved QAQC excel book to MongoDB===========")
        # Create excel book for qNTA if present
        if self.parameters["do_qnta"][1] == "yes":
            # Save project name to MongoDB using jobid
            self.gridfs.put(project_name, _id=f"{self.jobid}_project_name_qNTA", encoding="utf-8")
            # Save results excel file to MongoDB using id
            id = self.jobid + "_excel_qNTA"
            self.gridfs.put(task_fun.create_excel_book(self.qnta_map, chem_res=False), _id=id)
            self.log_memory_usage("create_excel_book from qnta_map")
            logger.info("===========Saved qNTA excel book to MongoDB===========")

    def save_QAQC_csv_to_mongo(self):
        in_memory_buffer = io.StringIO()

        # TODO check for presence of neg and pos mode detection statistics in the datamap
        if "All Detection Statistics (Pos)" in self.data_map and "All Detection Statistics (Neg)" in self.data_map:
            newdf = pd.concat(
                [self.data_map["All Detection Statistics (Pos)"], self.data_map["All Detection Statistics (Neg)"]],
                ignore_index=True,
            )
        elif "All Detection Statistics (Pos)" in self.data_map:
            newdf = self.data_map["All Detection Statistics (Pos)"]
        elif "All Detection Statistics (Neg)" in self.data_map:
            newdf = self.data_map["All Detection Statistics (Neg)"]

        # TODO convert the contatenated dataframe into a csv
        newdf.to_csv(in_memory_buffer, index=False)

        csv_data = in_memory_buffer.getvalue()

        # Save csv file to MongoDB using id
        id = self.jobid + "_csv_for_QAQC_visuals"
        self.gridfs.put(csv_data.encode(), _id=id)

        # Clear the buffer to free memory
        in_memory_buffer.close()
        # Delete the temporary dataframe/csv to free memory
        del newdf
        del csv_data
        # Prompt garbage collection
        gc.collect()

    def save_tripod_csv_to_mongo(self):
        in_memory_buffer = io.StringIO()

        # get columns needed for hazard/metadata vis, and drop duplicate columns
        cols_for_tripod_vis = [
            "Feature ID",
            "Mass",
            "Retention Time",
            "DTXCID_INDIVIDUAL_COMPONENT",
            "PATENT_COUNT_COLLAPSED",
            "PATENT_COUNT_COLLAPSED_NORM",
            "LITERATURE_COUNT_COLLAPSED",
            "LITERATURE_COUNT_COLLAPSED_NORM",
            "PUBMED_COUNT_COLLAPSED",
            "PUBMED_COUNT_COLLAPSED_NORM",
            "SOURCE_COUNT_COLLAPSED",
            "SOURCE_COUNT_COLLAPSED_NORM",
            "POSITIVE_MODE_AMENABILITY_PREDICTION",
            "NEGATIVE_MODE_AMENABILITY_PREDICTION",
        ]

        hazard_cols = [
            "Acute Mammalian Toxicity Oral_authority_mapped",
            "Acute Mammalian Toxicity Inhalation_authority_mapped",
            "Acute Mammalian Toxicity Dermal_authority_mapped",
            "Carcinogenicity_authority_mapped",
            "Genotoxicity Mutagenicity_authority_mapped",
            "Endocrine Disruption_authority_mapped",
            "Reproductive_authority_mapped",
            "Developmental_authority_mapped",
            "Neurotoxicity Repeat Exposure_authority_mapped",
            "Neurotoxicity Single Exposure_authority_mapped",
            "Systemic Toxicity Repeat Exposure_authority_mapped",
            "Systemic Toxicity Single Exposure_authority_mapped",
            "Skin Sensitization_authority_mapped",
            "Skin Irritation_authority_mapped",
            "Eye Irritation_authority_mapped",
            "Acute Aquatic Toxicity_authority_mapped",
            "Chronic Aquatic Toxicity_authority_mapped",
            "Persistence_authority_mapped",
            "Bioaccumulation_authority_mapped",
            "Exposure_authority_mapped",
            "Acute Mammalian Toxicity Oral_score_mapped",
            "Acute Mammalian Toxicity Inhalation_score_mapped",
            "Acute Mammalian Toxicity Dermal_score_mapped",
            "Carcinogenicity_score_mapped",
            "Genotoxicity Mutagenicity_score_mapped",
            "Endocrine Disruption_score_mapped",
            "Reproductive_score_mapped",
            "Developmental_score_mapped",
            "Neurotoxicity Repeat Exposure_score_mapped",
            "Neurotoxicity Single Exposure_score_mapped",
            "Systemic Toxicity Repeat Exposure_score_mapped",
            "Systemic Toxicity Single Exposure_score_mapped",
            "Skin Sensitization_score_mapped",
            "Skin Irritation_score_mapped",
            "Eye Irritation_score_mapped",
            "Acute Aquatic Toxicity_score_mapped",
            "Chronic Aquatic Toxicity_score_mapped",
            "Persistence_score_mapped",
            "Bioaccumulation_score_mapped",
            "Exposure_score_mapped",
            "Hazard Score",
            "Hazard Completeness Score",
        ]

        if self.parameters["search_hcd"][1] == "yes":
            for col in hazard_cols:
                cols_for_tripod_vis.append(col)

        newdf = self.data_map["Chemical Results"][cols_for_tripod_vis].drop_duplicates(
            subset=["Feature ID", "DTXCID_INDIVIDUAL_COMPONENT"]
        )

        # Create a total norm column
        sumcols = [col for col in newdf.columns if "NORM" in col]
        newdf["STRUCTURE_TOTAL_NORM"] = newdf[sumcols].sum(axis=1)

        newdf.to_csv(in_memory_buffer, index=False)

        csv_data = in_memory_buffer.getvalue()

        # Save csv file to MongoDB using id
        id = self.jobid + "_csv_data_for_tripod_vis"
        self.gridfs.put(csv_data.encode(), _id=id)

        # Clear the buffer to free memory
        in_memory_buffer.close()
        # Delete the temporary dataframe/csv to free memory
        del newdf
        del csv_data
        # Prompt garbage collection
        gc.collect()

    def save_decision_tree_info_to_mongo(self):
        # This funciton is a result of NTAW-711 and is only required for the decision tree on the MS1 QED deploymet
        self.mongo_save(self.data_map["Analysis Parameters"], step="Analysis Parameters")
        if "All Detection Statistics (Pos)" in self.data_map:
            self.mongo_save(self.data_map["All Detection Statistics (Pos)"], step="All Detection Statistics (Pos)")
        if "All Detection Statistics (Neg)" in self.data_map:
            self.mongo_save(self.data_map["All Detection Statistics (Neg)"], step="All Detection Statistics (Neg)")
