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

# from . import functions_Universal_v3 as fn

# from . import toxpi
from .utilities import *  # connect_to_mongoDB, connect_to_mongo_gridfs, reduced_file, api_search_masses, api_search_formulas,

from . import task_functions as task_fun
from .WebApp_plotter import WebApp_plotter
import io
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger("nta_app.ms1")

# import seaborn as sns
try:
    import seaborn as sns
except ModuleNotFoundError:
    logger.error("Seaborn is not installed. Please run 'pip install seaborn' to install it.")


# os.environ['IN_DOCKER'] = "False" #for local dev - also see similar switch in tools/output_access.py
NO_DASK = False  # set this to true to run locally without test (for debug purposes)


def run_nta_dask(
    parameters,
    input_dfs,
    tracer_df=None,
    run_sequence_pos_df=None,
    run_sequence_neg_df=None,
    jobid="00000000",
    verbose=True,
):
    in_docker = os.environ.get("IN_DOCKER") != "False"
    mongo_address = os.environ.get("MONGO_SERVER")

    if NO_DASK:
        run_nta(
            parameters,
            input_dfs,
            tracer_df,
            run_sequence_pos_df,
            run_sequence_neg_df,
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
        mongo_address=None,
        jobid="00000000",
        verbose=True,
        in_docker=True,
    ):
        logger.info("Initializing NtaRun Task")
        logger.info("parameters= {}".format(parameters))
        self.parameters = parameters
        self.tracer_df = tracer_df
        self.tracer_dfs_out = None
        self.run_sequence_pos_df = run_sequence_pos_df
        self.run_sequence_neg_df = run_sequence_neg_df
        self.dfs = input_dfs
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
        self.tracer_map = {}
        self.occurrence_heatmap_map = {}
        self.cv_scatterplot_map = {}
        # self.data_dir = os.path.join(self.base_dir, 'data', self.jobid)
        # self.new_download_dir = os.path.join(self.data_dir, "new")
        self.step = "Started"  # tracks the current step (for fail messages)
        # os.mkdir(self.data_dir)
        # os.mkdir(self.new_download_dir)
        self.tracer_plots_out = []
        self.occurrence_heatmaps_out = []
        self.cv_scatterplots_out = []

    def execute(self):
        self.step = "Check for existence of required columns"
        # 0: check existence of "Ionization mode" column
        self.check_existence_of_ionization_mode_column(self.dfs)
        # 0: check existence of 'mass column'
        self.check_existence_of_mass_column(self.dfs)
        # 0: check for alternate spellings of 'Retention_Time' column
        self.check_retention_time_column(self.dfs)
        # 0: sort dataframe columns alphabetically
        self.dfs = [df.reindex(sorted(df.columns), axis=1) if df is not None else None for df in self.dfs]
        # 0: create a status in mongo
        self.set_status("Processing", create=True)
        # 0: create an analysis_parameters sheet
        self.create_analysis_parameters_sheet()
        # 1: drop duplicates and throw out void volume
        self.step = "Dropping duplicates"
        self.assign_id()
        self.pass_through_cols()
        self.filter_void_volume(float(self.parameters["minimum_rt"][1]))  # throw out features below this (void volume)
        self.filter_duplicates()
        if self.verbose:
            logger.info("Dropped duplicates.")
            logger.info("dfs.size(): {}".format(len(self.dfs)))
            if self.dfs[0] is not None:
                logger.info("POS df length: {}".format(len(self.dfs[0])))
                logger.info("POS df columns: {}".format(self.dfs[0].columns))
            if self.dfs[1] is not None:
                logger.info("NEG df length: {}".format(len(self.dfs[1])))
                logger.info("NEG df columns: {}".format(self.dfs[1].columns))

        # 2: statistics
        self.step = "Calculating statistics"
        self.calc_statistics()
        if self.verbose:
            logger.info("Calculated statistics.")
            if self.dfs[0] is not None:
                logger.info("POS df length: {}".format(len(self.dfs[0])))
            if self.dfs[1] is not None:
                logger.info("NEG df length: {}".format(len(self.dfs[1])))

        # 2.1: Occurrence heatmap
        self.step = "Create heatmap"
        self.occurrence_heatmap(self.dfs)

        # 3: check tracers (optional)
        self.step = "Checking tracers"
        if self.verbose:
            logger.info("Checking tracers: DF check for debugging.")
            logger.info("dfs.size(): {}".format(len(self.dfs)))
            if self.dfs[0] is not None:
                logger.info("POS df length: {}".format(len(self.dfs[0])))
                logger.info("POS df columns: {}".format(self.dfs[0].columns.tolist()))
            if self.dfs[1] is not None:
                logger.info("NEG df length: {}".format(len(self.dfs[1])))
                logger.info("NEG df columns: {}".format(self.dfs[1].columns.tolist()))
        self.check_tracers()
        if self.verbose:
            logger.info("Checked tracers.")

        # 3.1: CV Scatterplot
        self.step = "Create scatterplot"
        self.cv_scatterplot(self.dfs)

        # 4: clean features
        self.step = "Cleaning features"
        self.clean_features()

        if self.verbose:
            logger.info("Cleaned features.")
            if self.dfs[0] is not None:
                logger.info("POS df length: {}".format(len(self.dfs[0])))
            if self.dfs[1] is not None:
                logger.info("NEG df length: {}".format(len(self.dfs[1])))

        # 4.1: Merge detection count columns onto tracers for export
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
            logger.info("combined df length: {}".format(len(self.df_combined)))

        # 7: search dashboard
        if self.parameters["search_dsstox"][1] == "yes":
            self.step = "Searching dsstox database"
            self.perform_dashboard_search()
            if self.parameters["search_hcd"][1] == "yes":
                self.step = "Searching Cheminformatics Hazard Module database"
                self.perform_hcd_search()

        # 8: Store data to MongoDB
        self.step = "Storing data"
        self.store_data()

        # 9: set status to completed
        self.step = "Displaying results"
        self.set_status("Completed")
        logger.warning("MS1 job {}: Processing complete.".format(self.jobid))

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

        Args:
            input_dfs (list of pandas DataFrames): A list of dataframes to check.

        This function checks each dataframe in the input list for the presence of a 'Mass' or 'm/z' column. If either of these columns is found, it takes appropriate action based on the ionization mode. If neither column exists, it raises a ValueError.

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
        # Check for the existence of alternate spellings of 'Retention_Time' column in input dataframes and rename to "Retention_Time".

        for df in input_dfs:
            if df is not None:
                # Check to see if there is not the expected spelling of "Retention_Time" column
                # if 'Retention_Time' not in df.columns:
                # replace alternative capitalizations
                df.rename(
                    columns={
                        "Retention_time": "Retention_Time",
                        "RETENTION_TIME": "Retention_Time",
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

    def set_status(self, status, create=False):
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
        posts = self.mongo.posts
        time_stamp = datetime.utcnow()
        post_id = self.jobid + "_" + "status"
        posts.update_one(
            {"_id": post_id},
            {"$set": {"_id": post_id, "date": time_stamp, "error_info": e}},
            upsert=True,
        )

    def get_step(self):
        return self.step

    def assign_id(self):
        if self.dfs[0] is not None and self.dfs[1] is not None:
            self.dfs[0] = task_fun.assign_feature_id(self.dfs[0])
            self.dfs[1] = task_fun.assign_feature_id(self.dfs[1], start=len(self.dfs[0].index) + 1)
        elif self.dfs[0] is not None:
            self.dfs[0] = task_fun.assign_feature_id(self.dfs[0])
        else:
            self.dfs[1] = task_fun.assign_feature_id(self.dfs[1])
        return

    def pass_through_cols(self):
        self.pass_through = [task_fun.passthrucol(df)[0] if df is not None else None for df in self.dfs]
        self.dfs = [task_fun.passthrucol(df)[1] if df is not None else None for df in self.dfs]
        return

    def filter_void_volume(self, min_rt):
        self.dfs = [
            df.loc[df["Retention_Time"] > min_rt].copy() if df is not None else None
            for index, df in enumerate(self.dfs)
        ]
        return

    def filter_duplicates(self):
        ppm = self.parameters["mass_accuracy_units"][1] == "ppm"
        mass_accuracy = float(self.parameters["mass_accuracy"][1])
        rt_accuracy = float(self.parameters["rt_accuracy"][1])
        # Perform duplicate flagging functions
        self.dfs = [
            task_fun.duplicates(df, mass_accuracy, rt_accuracy, ppm) if df is not None else None for df in self.dfs
        ]
        return

    def calc_statistics(self):
        ppm = self.parameters["mass_accuracy_units"][1] == "ppm"
        mass_accuracy = float(self.parameters["mass_accuracy"][1])
        rt_accuracy = float(self.parameters["rt_accuracy"][1])
        mrl_multiplier = float(self.parameters["mrl_std_multiplier"][1])
        self.dfs = [task_fun.chunk_stats(df, mrl_multiplier) if df is not None else None for df in self.dfs]
        # print(pos_adducts_selected)
        pos_adducts_selected = self.parameters["pos_adducts"][1]
        logger.info("pos adducts list: {}".format(self.parameters["pos_adducts"]))
        # print(neg_adducts_selected)
        neg_adducts_selected = self.parameters["neg_adducts"][1]
        logger.info("neg adducts list: {}".format(self.parameters["neg_adducts"]))
        # print(neutral_losses_selected)
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
                self.dfs[0], self.pass_through[0]
            )
            self.data_map["All Detection Statistics (Neg)"] = task_fun.column_sort_DFS(
                self.dfs[1], self.pass_through[1]
            )
        elif self.dfs[0] is not None:
            self.data_map["All Detection Statistics (Pos)"] = task_fun.column_sort_DFS(
                self.dfs[0], self.pass_through[0]
            )
        else:
            self.data_map["All Detection Statistics (Neg)"] = task_fun.column_sort_DFS(
                self.dfs[1], self.pass_through[1]
            )
        return

    def cv_scatterplot(self, input_dfs):
        # Set defaults
        plt.rcdefaults()
        # Set title
        titleText = "CV vs. Abundance"
        # Get user input CV threshold, convert to float
        max_replicate_cv_value = self.parameters["max_replicate_cv"][1]
        max_replicate_cv_value = float(max_replicate_cv_value)

        # get dataframe 'All Detection Statistics (Pos)' if it exists else None
        dfPos = (
            self.data_map["All Detection Statistics (Pos)"]
            if "All Detection Statistics (Pos)" in self.data_map
            else None
        )
        # get dataframe 'All Detection Statistics (Neg)' if it exists else None
        dfNeg = (
            self.data_map["All Detection Statistics (Neg)"]
            if "All Detection Statistics (Neg)" in self.data_map
            else None
        )
        # get 'Tracer Detection Statistics' if it exists else None
        dfTracer = (
            self.data_map["Tracer Detection Statistics"] if "Tracer Detection Statistics" in self.data_map else None
        )
        # Add conditional; if tracer exists reformat
        if dfTracer is not None:
            tracers = dfTracer[["Observed Mass", "Observed Retention Time"]].copy()
            tracers.rename({"Observed Mass": "Mass"}, axis=1, inplace=True)
            tracers.rename({"Observed Retention Time": "Retention Time"}, axis=1, inplace=True)
            tracers["spike"] = 1
            logger.info("cv scatterplot tracers columns= {}".format(tracers.columns.values))
        # combine the two dataframes, ignore non-existing dataframes
        dfCombined = (
            pd.concat([dfPos, dfNeg], axis=0, ignore_index=True, sort=False)
            if dfPos is not None and dfNeg is not None
            else dfPos
            if dfPos is not None
            else dfNeg
            if dfNeg is not None
            else None
        )
        # Get sample headers
        all_headers = task_fun.parse_headers(dfCombined)
        non_samples = ["MRL"]
        sam_headers = [
            sublist[0][:-1]
            for sublist in all_headers
            if len(sublist) > 1
            if not any(x in sublist[0] for x in non_samples)
        ]
        # Isolate sample_groups from stats columns
        prefixes = ["Mean ", "Median ", "CV ", "STD ", "Detection Count ", "Detection Percentage "]
        sample_groups = [item for item in sam_headers if not any(x in item for x in prefixes)]
        # Find CV cols from df, subset cv_df from df
        cv_cols = ["CV " + col for col in sample_groups]
        cv_df = dfCombined[cv_cols]
        # Find CV cols from df, subset cv_df from df
        mean_cols = ["Mean " + col for col in sample_groups]
        mean_df = dfCombined[mean_cols]
        # Carry over Mass and Retention_Time
        cv_df["Mass"] = dfCombined["Mass"]
        cv_df["Retention Time"] = dfCombined["Retention Time"]
        # AC 2/8/2024 Get minimum and maximum abundance values of dataframe (mean columns) for the purposes of setting the x-axis range
        min_abundance_value = mean_df.min(numeric_only=True).min()
        max_abundance_value = mean_df.max(numeric_only=True).max()
        if (
            min_abundance_value == 0
        ):  # If minimum abundance value is zero, then set minimum limit to zero (to avoid log issues on zero)
            min_abundance_limit = 0
        else:
            min_abundance_limit = 10 ** math.floor(math.log10(min_abundance_value))
        max_abundance_limit = 10 ** math.ceil(math.log10(max_abundance_value))
        # Create list, define blank strings
        li = []
        blanks = ["MB1", "BLK", "Blank", "BLANK", "blank", "MB", "mb"]
        # Loop through sample groups
        for x in sample_groups:
            # Take each sample's CV and mean, store in dummy variable
            cv = "CV " + x
            mean = "Mean " + x
            dum = pd.concat([cv_df[cv], mean_df[mean]], axis=1)
            dum.rename({cv: "CV"}, axis=1, inplace=True)
            dum.rename({mean: "Mean"}, axis=1, inplace=True)
            dum["sample"] = x
            dum["Mass"] = cv_df["Mass"]
            dum["Retention Time"] = cv_df["Retention Time"]
            # Add sample type (blank or sample)
            if any(i in x for i in blanks):
                dum["type"] = "blank"
            else:
                dum["type"] = "sample"
            # Append to list
            li.append(dum)

        # Concatenate plot, drop NAs
        plot = pd.concat(li)
        plot.dropna(axis=0, subset=["CV", "Mean"], how="any", inplace=True)
        logger.info("cv scatterplot plot columns= {}".format(plot.columns.values))

        # Conditional for if tracers are present:
        if dfTracer is not None:
            # Merge df with tracers to get labels
            plot2 = pd.merge(plot, tracers, how="left", on=["Mass", "Retention Time"])
        else:
            # If tracer plot doesn't exist, still need to create a spike column that is empty
            plot["spike"] = ""
            plot2 = plot.copy()

        plot2.replace(np.nan, 0, inplace=True)
        # Define subplots, set height and width
        f, axes = plt.subplots(1, 2)
        f.set_figheight(5)
        f.set_figwidth(15)
        # Set palette
        palette = ["whitesmoke", "firebrick"]
        sns.set_palette(palette, 2)
        # Blank plot
        a = sns.scatterplot(
            data=plot2.loc[((plot2["type"] == "blank")), :].sort_values("spike"),
            x="Mean",
            y="CV",
            hue="spike",
            edgecolor="black",
            alpha=0.5,
            ax=axes[0],
        )
        # Add CV red dashed line
        a.axhline(
            y=max_replicate_cv_value,
            color="red",
            linestyle="dashed",
            linewidth=1.5,
            alpha=1,
        )
        a.text(
            max_abundance_limit / 5,
            max_replicate_cv_value + 0.1,
            "CV = {}".format(max_replicate_cv_value),
            ha="center",
            va="center_baseline",
            weight="bold",
            size=14,
        )
        # Perform occurrence counts above and below CV by sample type
        red_count = len(plot2.loc[((plot2["type"] == "blank") & (plot2["spike"] == 1)), "CV"])
        red_flag_count = sum(
            plot2.loc[((plot2["type"] == "blank") & (plot2["spike"] == 1)), "CV"] > max_replicate_cv_value
        )

        white_count = len(plot2.loc[((plot2["type"] == "blank") & (plot2["spike"] == 0)), "CV"])
        white_flag_count = sum(
            plot2.loc[((plot2["type"] == "blank") & (plot2["spike"] == 0)), "CV"] > max_replicate_cv_value
        )

        # Only generate legend if tracers are submitted -- THIS ISN'T TRUE RIGHT NOW
        legend = a.legend(title="Unfiltered Occurrences", fontsize=14, title_fontsize=16)
        # Set legend labels
        if dfTracer is not None:
            legend.get_texts()[0].set_text(f"unknowns ({white_flag_count} of {white_count} above line)")
            try:
                legend.get_texts()[1].set_text(
                    f"tracers ({red_flag_count} of {red_count} above line)"
                )  # If tracers are present, add secondary legend label
            except IndexError:  # If no tracers found in blanks, set alternate legend
                # legend.set_text("tracers 0 of 0 above line)")
                pass
        # Make it pretty
        frame = legend.get_frame()  # sets up for color, edge, and transparency
        frame.set_facecolor("lightgray")  # color of legend
        frame.set_edgecolor("black")  # edge color of legend
        frame.set_alpha(1)  # deals with transparency
        # Adjust axes labels
        axes[0].set_title(titleText + ": Blanks", fontsize=18, weight="bold")
        axes[0].set_xlabel("Mean Abundance", fontsize=14)
        axes[0].set_ylabel("CV", fontsize=14)
        axes[0].set_ylim(0, 2.5)
        axes[0].set_xlim(
            min_abundance_limit, max_abundance_limit
        )  # Set x-axis to scale based on the min/max data points
        axes[0].set(xscale="log")
        axes[0].set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        axes[0].tick_params(axis="both", which="both", labelsize=12)

        # Sample plot
        b = sns.scatterplot(
            data=plot2.loc[((plot2["type"] != "blank")), :].sort_values("spike"),
            x="Mean",
            y="CV",
            hue="spike",
            edgecolor="black",
            alpha=0.5,
            ax=axes[1],
        )
        # Add CV red dashed line
        b.axhline(
            y=max_replicate_cv_value,
            color="red",
            linestyle="dashed",
            linewidth=1.5,
            alpha=1,
        )
        b.text(
            max_abundance_limit / 5,
            max_replicate_cv_value + 0.1,
            "CV = {}".format(max_replicate_cv_value),
            ha="center",
            va="center_baseline",
            weight="bold",
            size=14,
        )
        # Perform occurrence counts above and below CV by sample type
        red_count = len(plot2.loc[((plot2["type"] != "blank") & (plot2["spike"] == 1)), "CV"])
        red_flag_count = sum(
            plot2.loc[((plot2["type"] != "blank") & (plot2["spike"] == 1)), "CV"] > max_replicate_cv_value
        )
        white_count = len(plot2.loc[((plot2["type"] != "blank") & (plot2["spike"] == 0)), "CV"])
        white_flag_count = sum(
            plot2.loc[((plot2["type"] != "blank") & (plot2["spike"] == 0)), "CV"] > max_replicate_cv_value
        )
        # Only generate legend if tracers are submitted -- THIS ISN'T TRUE RIGHT NOW
        legend = b.legend(title="Unfiltered Occurrences", fontsize=14, title_fontsize=16)
        # Set legend labels
        if dfTracer is not None:
            legend.get_texts()[0].set_text(f"unknowns ({white_flag_count} of {white_count} above line)")
            legend.get_texts()[1].set_text(
                f"tracers ({red_flag_count} of {red_count} above line)"
            )  # If tracers are present, add secondary legend label
        # Make it pretty
        frame = legend.get_frame()  # sets up for color, edge, and transparency
        frame.set_facecolor("lightgray")  # color of legend
        frame.set_edgecolor("black")  # edge color of legend
        frame.set_alpha(1)  # deals with transparency
        # Adjust axes labels
        axes[1].set_title(titleText + ": Non-blanks", fontsize=18, weight="bold")
        axes[1].set_xlabel("Mean Abundance", fontsize=14)
        axes[1].set_ylabel("CV", fontsize=14)
        axes[1].set_ylim(0, 2.5)
        axes[1].set_xlim(min_abundance_limit, max_abundance_limit)
        axes[1].set(xscale="log")
        axes[1].set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        axes[1].tick_params(axis="both", which="both", labelsize=12)
        # Convert the plot to a bytes-like object
        buffer = io.BytesIO()
        plt.savefig(buffer)
        buffer.seek(0)
        # Store in class variable
        self.cv_scatterplots_out.append(buffer.getvalue())
        # Map to outputs
        self.cv_scatterplot_map["cv_scatterplot"] = self.cv_scatterplots_out[0]
        project_name = self.parameters["project_name"][1]
        self.gridfs.put(
            "&&".join(self.cv_scatterplot_map.keys()),
            _id=self.jobid + "_cv_scatterplots",
            encoding="utf-8",
            project_name=project_name,
        )
        self.mongo_save(self.cv_scatterplot_map["cv_scatterplot"], step="cv_scatterplot")
        # reset plt
        plt.clf()

    def occurrence_heatmap(self, input_dfs):
        plt.rcdefaults()
        # Get user input CV and Replicate thresholds
        max_replicate_cv_value = self.parameters["max_replicate_cv"][1]
        min_replicate_hits_percent = self.parameters["min_replicate_hits"][1]
        min_replicate_blanks_hit_percent = self.parameters["min_replicate_hits_blanks"][1]
        MRL_mult = float(self.parameters["mrl_std_multiplier"][1])
        # convert max_replicate_cv_value to a numeric value
        max_replicate_cv_value = pd.to_numeric(self.parameters["max_replicate_cv"][1], errors="coerce")
        # convert min_replicate_hits_percent to a numeric value
        min_replicate_hits_percent = pd.to_numeric(self.parameters["min_replicate_hits"][1], errors="coerce")
        # convert min_replicate_blanks_hits_percent to a numeric value
        min_replicate_blanks_hits_percent = pd.to_numeric(
            self.parameters["min_replicate_hits_blanks"][1], errors="coerce"
        )
        # get dataframe 'All Detection Statistics (Pos)' if it exists else None
        dfPos = (
            self.data_map["All Detection Statistics (Pos)"]
            if "All Detection Statistics (Pos)" in self.data_map
            else None
        )
        # get dataframe 'All Detection Statistics (Neg)' if it exists else None
        dfNeg = (
            self.data_map["All Detection Statistics (Neg)"]
            if "All Detection Statistics (Neg)" in self.data_map
            else None
        )
        # combine the two dataframes. Ignnore non-existing dataframes
        dfCombined = (
            pd.concat([dfPos, dfNeg], axis=0, ignore_index=True, sort=False)
            if dfPos is not None and dfNeg is not None
            else dfPos
            if dfPos is not None
            else dfNeg
            if dfNeg is not None
            else None
        )

        # Get sample headers
        all_headers = task_fun.parse_headers(dfCombined)
        non_samples = ["MRL"]
        sam_headers = [
            sublist[0][:-1]
            for sublist in all_headers
            if len(sublist) > 1
            if not any(x in sublist[0] for x in non_samples)
        ]
        # Isolate sample_groups from stats columns
        prefixes = ["Mean ", "Median ", "CV ", "STD ", "Detection Count ", "Detection Percentage "]
        sample_groups = [item for item in sam_headers if not any(x in item for x in prefixes)]
        logger.info("sample_groups= {}".format(sample_groups))
        # Blank_MDL - need to check what the blank samples are actually named
        blank_strings = ["MB", "Mb", "mb", "BLANK", "Blank", "blank", "BLK", "Blk"]
        blank_col = [item for item in sample_groups if any(x in item for x in blank_strings)]
        logger.info("blank_col= {}".format(blank_col))
        blank_mean = "Mean " + blank_col[0]
        blank_std = "STD " + blank_col[0]
        # AC Add blank replicate percentage column grab NTAW574
        blank_rper = "Detection Percentage " + blank_col[0]

        # Calculate MDL
        # AC 6/18/2024: Need to pull in MRL multiplier for MRL calculation
        dfCombined["MDL"] = dfCombined[blank_mean] + MRL_mult * dfCombined[blank_std]
        dfCombined["MDL"] = dfCombined["MDL"].fillna(dfCombined[blank_mean])
        dfCombined["MDL"] = dfCombined["MDL"].fillna(0)
        # AC Where blank replicate percentage column fails, zero out MDL - NTAW574
        dfCombined.loc[dfCombined[blank_rper] < min_replicate_blanks_hits_percent, "MDL"] = 0
        # Find CV, Rep_Percent, and Mean cols from df
        cv_cols = ["CV " + col for col in sample_groups]
        rper_cols = ["Detection Percentage " + col for col in sample_groups]
        mean_cols = ["Mean " + col for col in sample_groups]
        # Subset CV cols from df
        cv_df = dfCombined[cv_cols]

        # Get number of occurrences from the CV dataframe
        titleText = (
            "Heatmap of Feature Occurrences (n = "
            + str(cv_df.size)
            + ")\nSample Rep. Threshold = {}%; Blank Rep. Threshold = {}%; CV Threshold = {}; MRL Multiplier = {}".format(
                min_replicate_hits_percent, min_replicate_blanks_hits_percent, max_replicate_cv_value, MRL_mult
            )
        )

        # Blank out cvs in samples with <2 samples
        for x, y, z in zip(cv_cols, rper_cols, mean_cols):
            # Replace cv_df values with nan in cv_col for n_abun and MDL cutoffs
            # Check if replicate column is the blank column to determine which filter to apply - NTAW574
            if y == blank_rper:
                cv_df.loc[dfCombined[y] < min_replicate_blanks_hits_percent, x] = np.nan
            else:
                cv_df.loc[dfCombined[y] < min_replicate_hits_percent, x] = np.nan
            cv_df.loc[dfCombined[y] < min_replicate_hits_percent, x] = np.nan
            cv_df.loc[dfCombined[z] <= dfCombined["MDL"], x] = np.nan
        # Add sum of Trues for condition applied to cv dataframe
        cv_df["below count"] = (cv_df <= max_replicate_cv_value).sum(axis=1)
        # Sort values by how many detects are present
        cv_df = cv_df.sort_values("below count")
        # Remove below count column
        del cv_df[cv_df.columns[-1]]
        # Create masks for CV cutoffs
        above = cv_df > max_replicate_cv_value
        below = cv_df <= max_replicate_cv_value
        nan_ = cv_df.isna()
        # Use masks to changes values in cv_df to 1, 0, -1
        dum = np.where(above, 1, cv_df)
        dum = np.where(below, 0, dum)
        dum = np.where(nan_, -1, dum)
        # Create matrix from discretized dataframe
        cv_df_discrete = pd.DataFrame(dum, index=cv_df.index, columns=[col[3:] for col in cv_df.columns])
        cv_df_trans = cv_df_discrete.transpose()
        # Set Figure size and title
        plt.figure(figsize=(40, 15))
        plt.title(titleText, fontsize=40, pad=30, linespacing=1.5)
        # Create custom color mapping
        myColors = ((0.8, 0.8, 0.8, 1.0), (1.0, 1.0, 1.0, 1.0), (1, 0.0, 0.2, 1.0))
        cmap = LinearSegmentedColormap.from_list("Custom", myColors, len(myColors))
        # Plot heatmap
        ax = sns.heatmap(cv_df_trans, cmap=cmap, cbar_kws={"shrink": 0.2, "pad": 0.01})
        ax.set_ylabel("Sample Set", fontsize=28)
        ax.set_xlabel("Feature ID (n = " + str(len(cv_df)) + ")", fontsize=28)
        ax.set(xticklabels=[])
        ax.tick_params(axis="y", which="both", labelsize=24, labelrotation=0)
        # Add outside border
        ax.patch.set_edgecolor("black")
        ax.patch.set_linewidth(2)
        # Manually specify colorbar labelling after it's been generated
        colorbar = ax.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=32)
        colorbar.set_ticks([-0.667, 0, 0.667])
        colorbar.set_ticklabels(
            [
                "no occurrence ({})".format(nan_.sum().sum()),
                "CV <= {} ({})".format(max_replicate_cv_value, below.sum().sum()),
                "CV > {} ({})".format(max_replicate_cv_value, above.sum().sum()),
            ]
        )
        # Convert the plot to a bytes-like object
        buffer = io.BytesIO()
        plt.savefig(buffer)
        buffer.seek(0)
        # Store in class variable
        self.occurrence_heatmaps_out.append(buffer.getvalue())
        # Map to outputs
        self.occurrence_heatmap_map["occurrence_heatmap"] = self.occurrence_heatmaps_out[0]
        project_name = self.parameters["project_name"][1]
        self.gridfs.put(
            "&&".join(self.occurrence_heatmap_map.keys()),
            _id=self.jobid + "_occurrence_heatmaps",
            encoding="utf-8",
            project_name=project_name,
        )
        self.mongo_save(self.occurrence_heatmap_map["occurrence_heatmap"], step="occurrence_heatmap")
        # reset plt
        plt.clf()

    def check_tracers(self):
        if self.tracer_df is None:
            logger.info("No tracer file, skipping this step.")
            return
        if self.verbose:
            logger.info("Tracer file found, checking tracers.")
        ppm = self.parameters["mass_accuracy_units_tr"][1] == "ppm"
        mass_accuracy_tr = float(self.parameters["mass_accuracy_tr"][1])
        yaxis_scale = self.parameters["tracer_plot_yaxis_format"][1]
        trendline_shown = self.parameters["tracer_plot_trendline"][1] == "yes"

        # logger.info("check_tracers df[1]= {}".format(df[1].columns.tolist()))
        # logger.info("check_tracers self.tracer_df= {}".format(self.tracer_df.columns.tolist()))

        self.tracer_dfs_out = [
            (
                task_fun.check_feature_tracers(
                    df,
                    self.tracer_df,
                    mass_accuracy_tr,
                    float(self.parameters["rt_accuracy_tr"][1]),
                    ppm,
                )[0]
                if df is not None
                else None
            )
            for index, df in enumerate(self.dfs)
        ]
        self.dfs = [
            (
                task_fun.check_feature_tracers(
                    df,
                    self.tracer_df,
                    mass_accuracy_tr,
                    float(self.parameters["rt_accuracy_tr"][1]),
                    ppm,
                )[1]
                if df is not None
                else None
            )
            for index, df in enumerate(self.dfs)
        ]

        self.tracer_dfs_out = [format_tracer_file(df) if df is not None else None for df in self.tracer_dfs_out]

        # declare plotter
        df_WA = WebApp_plotter()

        # plot
        if self.tracer_dfs_out[0] is not None:
            listOfPNGs, df_debug, debug_list = df_WA.make_seq_scatter(
                # data_path='./input/summary_tracer.xlsx',
                df_in=self.tracer_dfs_out[0],
                seq_csv=self.run_sequence_pos_df,
                ionization="pos",
                y_scale=yaxis_scale,
                fit=trendline_shown,
                share_y=False,
                # y_fixed=False,
                y_step=6,
                same_frame=False,
                legend=True,
                # =None,
                dark_mode=False,
            )

            self.tracer_plots_out.append(listOfPNGs)

        else:
            self.tracer_plots_out.append(None)

        # declare plotter
        df_WA = WebApp_plotter()

        # plot
        if self.tracer_dfs_out[1] is not None:
            logger.info("self.tracer_dfs_out[1] shape= {}".format(self.tracer_dfs_out[1].shape))
            logger.info("self.tracer_dfs_out[1] columns= {}".format(self.tracer_dfs_out[1].columns.values))

            listOfPNGs, df_debug, debug_list = df_WA.make_seq_scatter(
                # data_path='./input/summary_tracer.xlsx',
                df_in=self.tracer_dfs_out[1],
                seq_csv=self.run_sequence_neg_df,
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
            logger.info("df_debug columns= {}".format(df_debug.columns.values))

        else:
            self.tracer_plots_out.append(None)

        # implements part of NTAW-143
        dft = pd.concat([self.tracer_dfs_out[0], self.tracer_dfs_out[1]])
        if self.pass_through[0] is not None and self.pass_through[1] is not None:
            passthru = pd.concat([self.pass_through[0], self.pass_through[1]])
        elif self.pass_through[0] is not None:
            passthru = self.pass_through[0]
        else:
            passthru = self.pass_through[1]

        logger.info("tracer dft= {}".format(dft.columns.tolist()))
        logger.info("tracer passthru= {}".format(passthru.columns.tolist()))
        self.data_map["Tracer Detection Statistics"] = task_fun.column_sort_TSR(dft, passthru)

        if self.tracer_plots_out[0] is not None:
            for i in range(len(self.tracer_plots_out[0])):
                self.tracer_map["tracer_plot_pos_" + str(i + 1)] = self.tracer_plots_out[0][i]

        # Add an if statement below to account for: if only negative mode data is entered, and only a negative tracer file is submitted, tracer_plots_out will only have one entry at [0]
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

    def clean_features(self):
        controls = [
            float(self.parameters["min_replicate_hits"][1]),
            float(self.parameters["max_replicate_cv"][1]),
            float(self.parameters["min_replicate_hits_blanks"][1]),
            float(self.parameters["mrl_std_multiplier"][1]),
        ]
        tracer_df_bool = False
        if self.tracer_df is not None:
            tracer_df_bool = True
        self.docs = [
            task_fun.clean_features(df, controls, tracer_df=tracer_df_bool)[1] if df is not None else None
            for index, df in enumerate(self.dfs)
        ]
        self.dfs_flagged = [
            task_fun.clean_features(df, controls, tracer_df=tracer_df_bool)[2] if df is not None else None
            for index, df in enumerate(self.dfs)
        ]
        self.dfs = [
            task_fun.clean_features(df, controls, tracer_df=tracer_df_bool)[0] if df is not None else None
            for index, df in enumerate(self.dfs)
        ]
        # subtract blanks from means
        self.dfs = [task_fun.Blank_Subtract_Mean(df) if df is not None else None for index, df in enumerate(self.dfs)]
        # subtract blanks from means
        self.dfs_flagged = [
            task_fun.Blank_Subtract_Mean(df) if df is not None else None for index, df in enumerate(self.dfs_flagged)
        ]
        return

    def merge_columns_onto_tracers(self):
        # Check for tracer file, if not present return
        if self.tracer_df is None:
            logger.info("No tracer file, skipping this step.")
            return
        # Grab the tracer dataframe from self.data_map
        dft = self.data_map["Tracer Detection Statistics"]
        logger.info("dft: {}".format(dft.columns))
        # create summary table
        if "DTXSID" not in dft.columns:
            dft["DTXSID"] = ""
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

        self.data_map["Tracer Summary"] = dft
        return

    def create_flags(self):
        self.dfs = [task_fun.flags(df) if df is not None else None for df in self.dfs]

    def combine_modes(self):
        tracer_df_bool = False
        if self.tracer_df is not None:
            tracer_df_bool = True
        # combine dfs from both modes
        self.df_combined = task_fun.combine(self.dfs[0], self.dfs[1])
        # combine df_flaggeds from both modes
        self.df_flagged_combined = task_fun.combine(self.dfs_flagged[0], self.dfs_flagged[1])
        # combine docs from both modes
        self.doc_combined = task_fun.combine_doc(self.docs[0], self.docs[1], tracer_df=tracer_df_bool)

        # NTAW-577: Replace zero values of "Selected MRL" column with blank cells prior to exporting to data_map
        self.doc_combined["Selected MRL"] = self.doc_combined["Selected MRL"].replace(0, "")
        # Also replace zero values of blank mean columns with blank cells prior to exporting to data_map
        blanks = ["MB", "mb", "mB", "Mb", "blank", "Blank", "BLANK"]
        Mean = self.doc_combined.columns[self.doc_combined.columns.str.contains(pat="Mean ")].tolist()
        Mean_MB = [md for md in Mean if any(x in md for x in blanks)]
        self.doc_combined[Mean_MB] = self.doc_combined[Mean_MB].replace(0, "")

        self.data_map["Decision Documentation"] = self.doc_combined

        self.mpp_ready = task_fun.MPP_Ready(self.df_combined, self.pass_through, tracer_df_bool, flagged=False)
        self.mpp_ready_flagged = task_fun.MPP_Ready(
            self.df_flagged_combined, self.pass_through, tracer_df_bool, flagged=True
        )

        self.data_map["Final Occurrence Matrix"] = reduced_file(self.mpp_ready)
        self.data_map["Final Occurrence Matrix (flags)"] = reduced_file(self.mpp_ready_flagged)

    def perform_dashboard_search(self, lower_index=0, upper_index=None, save=True):
        logging.info(
            "Rows flagged for dashboard search: {} out of {}".format(
                len(self.df_flagged_combined.loc[self.df_flagged_combined["For_Dashboard_Search"] == "1", :]),
                len(self.df_flagged_combined),
            )
        )
        to_search = self.df_flagged_combined.loc[
            self.df_flagged_combined["For_Dashboard_Search"] == "1", :
        ].copy()  # only rows flagged
        if self.parameters["search_mode"][1] == "mass":
            to_search.drop_duplicates(subset="Mass", keep="first", inplace=True)
        else:
            # NTAW-94 rename compound to formula
            to_search.drop_duplicates(subset="Formula", keep="first", inplace=True)
        n_search = len(to_search)  # number of fragments to search
        logger.info("Total # of queries: {}".format(n_search))

        to_search = to_search.iloc[lower_index:upper_index, :]
        if self.parameters["search_mode"][1] == "mass":
            mono_masses = task_fun.masses(to_search)
            dsstox_search_df = api_search_masses_batch(
                mono_masses,
                float(self.parameters["parent_ion_mass_accuracy"][1]),
                batchsize=150,
                jobid=self.jobid,
            )
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
        dsstox_search_df = self.mpp_ready_flagged[["Feature ID", "Mass", "Retention Time"]].merge(
            dsstox_search_df, how="right", left_on="Mass", right_on="INPUT"
        )

        # Calculate toxcast_percent_active values
        dsstox_search_df = task_fun.calc_toxcast_percent_active(dsstox_search_df)

        self.data_map["Chemical Results"] = dsstox_search_df
        self.search_results = dsstox_search_df

    def perform_hcd_search(self):
        logger.info(f"Querying the HCD with DTXSID identifiers")
        if len(self.search_results) > 0:
            dtxsid_list = self.search_results["DTXSID"].unique()
            hcd_results = batch_search_hcd(dtxsid_list)
            self.search_results = self.search_results.merge(hcd_results, how="left", on="DTXSID")
            self.data_map["Chemical Results"] = self.search_results
            self.data_map["hcd_search"] = hcd_results

    def store_data(self):
        logger.info(f"Storing data files to MongoDB")
        project_name = self.parameters["project_name"][1]
        self.gridfs.put(
            "&&".join(self.data_map.keys()),
            _id=self.jobid + "_file_names",
            encoding="utf-8",
            project_name=project_name,
        )
        for key in self.data_map.keys():
            self.mongo_save(self.data_map[key], step=key)

    def mongo_save(self, file, step=""):
        if isinstance(file, pd.DataFrame):
            to_save = file.to_json(orient="split")
        else:
            to_save = file
        id = self.jobid + "_" + step

        project_name = self.parameters["project_name"][1]
        self.gridfs.put(to_save, _id=id, encoding="utf-8", project_name=project_name)
