import pymongo as pymongo
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import gridfs
import io
import os
import logging
import json
import math
import requests
import time
import numpy as np
import pandas as pd
from . import task_functions as task_fun

logger = logging.getLogger("nta_app.ms1")

DSSTOX_API = os.environ.get("UBERTOOL_REST_SERVER")
HCD_API = os.environ.get("HCD_API_URL_DEV")


def connect_to_mongoDB(address):
    """
    With address from environment, create Mongo Collection "nta_runs". All entries
    must have timestamp, items are held for 24 hours.

    Args:
        address (string; self.mongo = os.environ.get("MONGO_SERVER"))
    Returns:
        mongo_db (Mongo collection; storage space for NTA outputs)
    """
    mongo = pymongo.MongoClient(host=address)
    mongo_db = mongo["nta_runs"]
    mongo.nta_runs.Collection.create_index([("date", pymongo.DESCENDING)], expireAfterSeconds=86400)
    # ALL entries into mongo.nta_runs must have datetime.utcnow() timestamp, which is used to delete the record after 86400
    # seconds, 24 hours.
    return mongo_db


def connect_to_mongo_gridfs(address):
    """
    Using address from environment, connect to MongoDB and return gridfs storage structure.

    Args:
        address (string; self.mongo = os.environ.get("MONGO_SERVER"))
    Returns:
        fs (Mongo gridded storage)
    """
    db = pymongo.MongoClient(host=address).nta_storage
    print("Connecting to mongodb at {}".format(address))
    fs = gridfs.GridFS(db)
    return fs


# # function to remove columns from a given dataframe, df_in. The columns to be removed are determined by the
# # a given list of strings.
# def remove_columns(df_in, list_of_columns2remove):
#     """


#     Args:
#         None
#     Returns:
#         None
#     """
#     df = df_in.copy()
#     df.drop(list_of_columns2remove, axis=1, inplace=True)
#     return df


def reduced_file(df_in):
    """
    Take dataframe, and remove unnecessary columns in preparation for output.

    Args:
        df_in (Pandas dataframe)
    Returns:
        df (Pandas dataframe)
    """
    df = df_in.copy()
    headers = task_fun.parse_headers(df)
    keeps_str = ["MB_", "blank", "blanks", "BLANK", "Blank", "Mean", "Sub"]
    to_drop = [
        item for sublist in headers for item in sublist if (len(sublist) > 1) & (not any(x in item for x in keeps_str))
    ]
    to_drop.extend(df.columns[(df.columns.str.contains(pat="CV |Detection Count |Median |STD ") == True)].tolist())
    to_drop.extend(
        df.columns[
            (df.columns.str.contains(pat="Mean ") == True)
            & (df.columns.str.contains(pat="MB|blank|blanks|BLANK|Blank|Sub") == False)
        ].tolist()
    )
    if "Mean_ALLMB" in df.columns.values.tolist():
        to_drop.extend(["Mean_ALLMB"])
    df.drop(to_drop, axis=1, inplace=True)
    return df


def response_log_wrapper(api_name: str):
    def api_log_decorator(request_func):
        def wrapper(*args, **kwargs):
            logger.info(f"============ calling REST API: {api_name}")
            start_time = time.perf_counter()
            response = request_func(*args, **kwargs)
            logger.info(f"Response: {response}   Run time: {time.perf_counter() - start_time}")
            return response

        return wrapper

    return api_log_decorator


@response_log_wrapper("DSSTOX")
def api_search_masses(masses, accuracy, jobid="00000"):
    """
    Formats list of masses for searching into JSON and using the DSSTOX_API
    set in os.environ.get("UBERTOOL_REST_SERVER"), return a POST request.

    Args:
        masses (list of floats)
        accuracy (integer; assumes ppm units)
        jobid (string)
    Returns:
        POST request of the formatted masses
    """
    input_json = json.dumps({"search_by": "mass", "query": masses, "accuracy": accuracy})  # assumes ppm

    api_url = "{}/rest/ms1/batch/{}".format(DSSTOX_API, jobid)

    logger.info(api_url)
    http_headers = {"Content-Type": "application/json"}
    return requests.post(api_url, headers=http_headers, data=input_json)


def api_search_masses_batch(masses, accuracy, batchsize=50, jobid="00000"):
    """
    Function for passing 50 masses at a time to the above api_search_masses() function.

    Args:
        masses (list of floats)
        accuracy (integer; assumes ppm units)
        batchsize (int; default is 50)
        jobid (string)
    Returns:
        POST request of the formatted masses
    """
    n_masses = len(masses)
    logging.info("Sending {} masses in batches of {}".format(n_masses, batchsize))
    i = 0
    while i < n_masses:
        end = i + batchsize - 1
        if end > n_masses - 1:
            end = n_masses - 1
        response = api_search_masses(masses[i : end + 1], accuracy, jobid)
        if not response.ok:  # check if we got a successful response
            raise requests.exceptions.HTTPError(
                "Unable to access DSSTOX API. Please contact an administrator or try turning the DSSTox search option off."
            )
        dsstox_search_json = io.StringIO(json.dumps(response.json()["results"]))  # can be an empty string if no hits
        if i == 0:
            dsstox_search_df = pd.read_json(
                dsstox_search_json,
                orient="split",
                dtype={"TOXCAST_NUMBER_OF_ASSAYS/TOTAL": "object"},
            )
        else:
            new_search_df = pd.read_json(
                dsstox_search_json,
                orient="split",
                dtype={"TOXCAST_NUMBER_OF_ASSAYS/TOTAL": "object"},
            )
            dsstox_search_df = pd.concat(
                [dsstox_search_df, new_search_df], ignore_index=True
            )  # Added ignore index, may not be needed 11/2 MWB
        i = i + batchsize

    return dsstox_search_df


@response_log_wrapper("DSSTOX")
def api_search_formulas(formulas, jobID="00000"):
    """
    Formats list of formulae for searching into JSON and using the DSSTOX_API
    set in os.environ.get("UBERTOOL_REST_SERVER"), return a POST request.

    Args:
        masses (list of strings)
        accuracy (integer; assumes ppm units)
        jobid (string)
    Returns:
        POST request of the formatted strings
    """
    input_json = json.dumps({"search_by": "formula", "query": formulas})  # assumes ppm
    if "edap-cluster" in DSSTOX_API:
        api_url = "{}/rest/ms1/batch/{}".format(DSSTOX_API, jobID)
    else:
        api_url = "{}/nta/rest/ms1/batch/{}".format(DSSTOX_API, jobID)
    logger.info(api_url)
    http_headers = {"Content-Type": "application/json"}
    return requests.post(api_url, headers=http_headers, data=input_json)


@response_log_wrapper("HCD")
def api_search_hcd(dtxsid_list):
    """
    Formats list of DTXSIDs for searching into JSON and using the HCD_API
    set in os.environ.get("HCD_API_URL_DEV"), return a POST request.

    Args:
        dtxsid_list (list of strings)
    Returns:
        POST request of the formatted masses
    """
    post_data = {
        "chemicals": [],
        "options": {"noRecords": "true", "usePredictions": "true"},
    }
    headers = {"content-type": "application/json"}
    url = HCD_API
    for dtxsid in dtxsid_list:
        post_data["chemicals"].append({"chemical": {"sid": dtxsid}})
    return requests.post(url, data=json.dumps(post_data), headers=headers)


def batch_search_hcd(dtxsid_list, batchsize=200):
    """
    Function for passing 200 DTXSIDs at a time to the above api_search_hcd() function.

    Args:
        dtxsid_list (list of strings)
        batchsize (int; default is 200)
    Returns:
        POST request of the formatted masses
    """
    result_dict = {}
    logger.info(f"Search {len(dtxsid_list)} DTXSIDs in HCD")
    for i in range(0, len(dtxsid_list), batchsize):
        logger.info(f"HCD Query: {i//batchsize} of {len(dtxsid_list)//batchsize} batches")
        response = api_search_hcd(dtxsid_list[i : i + batchsize])
        chem_data_list = json.loads(response.content)["hazardChemicals"]
        for chemical in chem_data_list:
            chemical_id = chemical["chemicalId"].split("|")[0]
            result_dict[chemical_id] = {}
            for data in chemical["scores"]:
                result_dict[chemical_id][f'{data["hazardName"]}_score'] = data["finalScore"]
                result_dict[chemical_id][f'{data["hazardName"]}_authority'] = (
                    data["finalAuthority"] if "finalAuthority" in data.keys() else ""
                )
    return pd.DataFrame(result_dict).transpose().reset_index().rename(columns={"index": "DTXSID"})


def format_tracer_file(df_in):
    """
    Function for formatting the Tracer Detection Statistics dataframe(s). Calculates and inserts
    two columns, "Mass Error (PPM)" and "Retention Time Difference", then returns the dataframe.

    Args:
        df_in (Pandas dataframe)
    Returns:
        df (dataframe with two additional columns, Mass Error (PPM) and Retention Time Difference)
    """
    df = df_in.copy()
    rt_diff = df["Observed Retention Time"] - df["Retention_Time"]
    mass_diff = ((df["Observed Mass"] - df["Monoisotopic_Mass"]) / df["Monoisotopic_Mass"]) * 1000000
    df.insert(7, "Mass Error (PPM)", mass_diff)
    df.insert(9, "Retention Time Difference", rt_diff)
    return df


# def create_tracer_plot(df_in):
#     """


#     Args:
#         None
#     Returns:
#         None
#     """
#     mpl_logger = logging.getLogger("matplotlib")
#     mpl_logger.setLevel(logging.WARNING)
#     headers = task_fun.parse_headers(df_in)
#     abundance = [item for sublist in headers for item in sublist if len(sublist) > 1]
#     fig, ax = plt.subplots()
#     for i, tracer in df_in.iterrows():
#         y = tracer[abundance]
#         x = abundance
#         ax.plot(x, y, marker="o", label=tracer[0])
#         ax.set_ylabel("Log abundance")
#         ax.set_xlabel("Sample name")
#     plt.yscale("log")
#     plt.xticks(rotation=-90)
#     plt.legend()
#     plt.tight_layout()
#     sf = ScalarFormatter()
#     sf.set_scientific(False)
#     ax.yaxis.set_major_formatter(sf)
#     ax.margins(x=0.3)
#     buffer = io.BytesIO()
#     plt.savefig(buffer)
#     plt.close()
#     return buffer.getvalue()
