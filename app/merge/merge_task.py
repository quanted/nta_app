import pandas as pd
import os
import io
import csv
import time
import logging
import traceback
import shutil
import json
from datetime import datetime
from dask.distributed import Client, LocalCluster, fire_and_forget
from django.urls import reverse
from .utilities import connect_to_mongoDB, connect_to_mongo_gridfs, make_hyperlink
from .merge_functions import process_MS2_data
from ...tools.ms2.send_email import send_ms2_finished
from openpyxl.utils import get_column_letter

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
        self.parameters = parameters
        self.project_name = parameters["project_name"][1]
        logger.info(f"\n============= Job ID: {jobid}")
        logger.info(input_data)
        self.input_ms1 = input_data["MS1"]
        # self.input_ms2 = [input_data["MS2_pos"], input_data["MS2_neg"]] # NTAW-158: Update how MS2 data is handled, needs to be a list of MS2 files / ignore this
        self.input_ms2 = input_data["MS2_pos"] + input_data["MS2_neg"]
        self.n_files = len(self.input_ms2)
        self.results_df = [None]
        self.results_link = results_link
        self.mass_accuracy_tolerance = float(parameters["mass_accuracy_tolerance"][1])
        self.rt_tolerance = float(parameters["rt_tolerance"][1])
        self.jobid = jobid
        self.verbose = verbose
        self.in_docker = in_docker
        self.mongo_address = mongo_address
        self.mongo = connect_to_mongoDB(self.mongo_address)
        self.gridfs = connect_to_mongo_gridfs(self.mongo_address)
        self.ms1_data_map = {}
        self.step = "Started"  # tracks the current step (for fail messages)

        self.create_analysis_parameters_sheet()

        self.ms1_data_map["chemical_results"] = (
            self.input_ms1 if isinstance(self.input_ms1, pd.DataFrame) else self.input_ms1
        )
        # self.ms1_data_map = (
        #     {"dsstox_search": self.input_ms1} if isinstance(self.input_ms1, pd.DataFrame) else self.input_ms1
        # )

        logger.info(f"\n============= Job ID: {jobid}")

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
            label = self.parameters[key][0]
            value = self.parameters[key][1]
            df_analysis_parameters.loc[len(df_analysis_parameters)] = [label, value]

        # add the dataframe to the data_map with the sheet name of 'Analysis Parameters'
        self.ms1_data_map["Analysis Parameters"] = df_analysis_parameters

        return

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
            # logger.info("Store file names")
            # self.gridfs.put(
            #     "&&".join(self.ms1_data_map.keys()),
            #     _id=self.jobid + "_file_names",
            #     encoding="utf-8",
            #     project_name=self.project_name,
            # )
            # logger.info("Store data to each file name")
            # for key in self.ms1_data_map.keys():
            #     self.mongo_save(self.ms1_data_map[key], data_name=key)

            logger.info(f"Merge Parameters: {self.parameters}")
            logger.info("Store results excel sheet to MongoDB")
            self.save_excel_to_mongo()

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

    def save_excel_to_mongo(self):
        # Create an excel sheet from the datamap and save it to MongoDB
        in_memory_buffer = io.BytesIO()
        # Deletes the old DTXSID hyperlink column, since this column is passed in as 'Null'
        self.ms1_data_map["chemical_results"] = self.ms1_data_map["chemical_results"].drop("CompTox links", axis=1)
        # Inserts a new CompTox links column
        self.ms1_data_map["chemical_results"].insert(
            loc=8,
            column="CompTox links",
            value=self.ms1_data_map["chemical_results"]["DTXSID"].apply(lambda x: make_hyperlink(x)),
        )

        # Get list of keys in the self.ms1_data_map dictionary
        keys_list = list(self.ms1_data_map.keys())

        # Convert self.ms1_data_map dictionary into an excel workbook
        with pd.ExcelWriter(in_memory_buffer, engine="openpyxl") as writer:
            workbook = writer.book
            for df_name, df in self.ms1_data_map.items():
                df.to_excel(writer, sheet_name=df_name, index=False)

                sheet_num = keys_list.index(df_name)
                sheet = workbook.worksheets[sheet_num]

                # Freeze the first row in the sheet
                sheet.freeze_panes = "A2"

                # Format each column width to fit the longest string contained within the column
                for column in df:
                    try:
                        column_width = max(df[column].astype(str).map(len).max(), len(column)) + 1
                        col_idx = df.columns.get_loc(column) + 1
                        col_letter = get_column_letter(col_idx)
                        sheet.column_dimensions[col_letter].width = column_width
                    except AttributeError:
                        pass

                if sheet_num == 1:
                    # Format DTXSID column hyperlinks
                    for i in range(sheet.max_row):
                        cell = sheet.cell(row=i + 2, column=9)
                        cell.style = "Hyperlink"
                    # Format decimal columns to scientific notation
                    for cell in sheet["P"]:
                        cell.number_format = "0.00E+00"
                    for cell in sheet["Y"]:
                        cell.number_format = "0.00E+00"
                    # Format extra long column widths
                    sheet.column_dimensions["I"].width = 18
                    sheet.column_dimensions["G"].width = 54
                    sheet.column_dimensions["J"].width = 54
                    sheet.column_dimensions["M"].width = 54
                    sheet.column_dimensions["P"].width = 18
                    sheet.column_dimensions["Y"].width = 18

        excel_data = in_memory_buffer.getvalue()
        # Save project name to MongoDB using jobid
        project_name = self.project_name
        self.gridfs.put(project_name, _id=f"{self.jobid}_project_name", encoding="utf-8")
        # Save results excel file to MongoDB using id
        id = self.jobid + "_merge_excel"
        self.gridfs.put(excel_data, _id=id)
