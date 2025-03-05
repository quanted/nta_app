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
# Retrieve API strings
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


def make_hyperlink(value, url="https://comptox.epa.gov/dashboard/chemical/details/{}"):
    """
    Function is used to display a URL as a hyperlink when the returned string is passed into an Excel cell.
    The hyperlink text will display the 'value' parameter.

    Args:
        value (string; dynamic part of the destination url)
        url (string, static part of the destination url with curly cr)
    Returns:
        The Excel hyperlink command.
    """
    return '=HYPERLINK("%s", "%s")' % (url.format(value), value)


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


def reduced_file(df_in, blank_headers, sample_headers):
    """
    Take dataframe, and remove unnecessary columns in preparation for output.

    Args:
        df_in (Pandas dataframe)
    Returns:
        df (Pandas dataframe)
    """
    # Copy dataframe
    df = df_in.copy()

    # Identify columns to drop
    sample_groups = blank_headers + sample_headers
    to_drop = [item for sublist in sample_groups for item in sublist] + ["MRL (3x)", "MRL (5x)", "MRL (10x)"]

    # Drop columns in list from df
    df.drop(to_drop, axis=1, inplace=True)
    # Return reduced df
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
    # Convert Python dictionary into JSON string
    input_json = json.dumps({"search_by": "mass", "query": masses, "accuracy": accuracy})  # assumes ppm
    # Get API url
    api_url = "{}/rest/ms1/batch/{}".format(DSSTOX_API, jobid)
    # Print API url to logger
    logger.info(api_url)
    http_headers = {"Content-Type": "application/json"}
    # Return POST request from API
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
    # Get length of masses list
    n_masses = len(masses)
    # Print to logger
    logger.info("===========Sending {} masses in batches of {}===========".format(n_masses, batchsize))
    # Get masses list into list of lists based on batchsize
    masses_li = [masses[i : i + batchsize] for i in range(0, len(masses), batchsize)]
    logger.info("===========Finished retrieving batches from DSSTox===========")
    # Iterate through masses in batchsize chunks, calling the api_search_masses() function.
    responses = [api_search_masses(x, accuracy, jobid) for x in masses_li]
    # Check responses ok, raise error if not
    response_checks = [x.ok for x in responses]
    response_checks = pd.Series(response_checks)
    if not response_checks.all():
        raise requests.exceptions.HTTPError(
            "Unable to access DSSTOX API. Please contact an administrator or try turning the DSSTox search option off."
        )
    else:
        logger.info("===========All responses are OK===========")
    # get json results as strings for responses from responses list
    dsstox_search_json = [io.StringIO(json.dumps(x.json()["results"])) for x in responses]
    logger.info("===========Finished converting JSON results to strings===========")
    # Read and concatenate json strings
    dsstox_search_df = pd.concat(
        [
            pd.read_json(x, orient="split", dtype={"TOXCAST_NUMBER_OF_ASSAYS/TOTAL": "object"})
            for x in dsstox_search_json
        ],
        ignore_index=True,
    )
    logger.info("===========Finished reading and concatenating JSON results===========")
    # Return dataframe
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
    # Convert Python dictionary into JSON string
    input_json = json.dumps({"search_by": "formula", "query": formulas})  # assumes ppm
    # Get API url
    if "edap-cluster" in DSSTOX_API:
        api_url = "{}/rest/ms1/batch/{}".format(DSSTOX_API, jobID)
    else:
        api_url = "{}/nta/rest/ms1/batch/{}".format(DSSTOX_API, jobID)
    # Print API url to logger
    logger.info(api_url)
    http_headers = {"Content-Type": "application/json"}
    # Return POST request from API
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
    # Format dictionary for data to POST
    post_data = {
        "chemicals": [],
        "options": {"noRecords": "true", "usePredictions": "true"},
    }
    headers = {"content-type": "application/json"}
    # Get API url
    url = HCD_API
    # Iterate through list of DTXSIDs, append to dictionary
    for dtxsid in dtxsid_list:
        post_data["chemicals"].append({"chemical": {"sid": dtxsid}})
    # Return POST request from API
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
    # Define dictionary for results
    result_dict = {}
    # Update logger
    logger.info(f"Search {len(dtxsid_list)} DTXSIDs in HCD")
    # Interate through dtxsid_list by batchsize
    for i in range(0, len(dtxsid_list), batchsize):
        # Update logger
        logger.info(f"HCD Query: {i//batchsize} of {len(dtxsid_list)//batchsize} batches")
        # Call api_search_hcd() for current chunk
        response = api_search_hcd(dtxsid_list[i : i + batchsize])
        # Convert JSON response to dictionary
        chem_data_list = json.loads(response.content)["hazardChemicals"]
        # Iterate through dictionary, format results
        for chemical in chem_data_list:
            chemical_id = chemical["chemicalId"].split("|")[0]
            result_dict[chemical_id] = {}
            for data in chemical["scores"]:
                result_dict[chemical_id][f'{data["hazardName"]}_score'] = data["finalScore"]
                result_dict[chemical_id][f'{data["hazardName"]}_authority'] = (
                    data["finalAuthority"] if "finalAuthority" in data.keys() else ""
                )
    # Return dataframe of dictionary data
    return pd.DataFrame(result_dict).transpose().reset_index().rename(columns={"index": "DTXSID"})
