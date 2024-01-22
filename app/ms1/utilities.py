
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
#from .functions_Universal_v3 import parse_headers

logger = logging.getLogger("nta_app.ms1")
logger.setLevel(logging.INFO)

DSSTOX_API = os.environ.get('UBERTOOL_REST_SERVER')
#DSSTOX_API = 'http://127.0.0.1:7777'


def connect_to_mongoDB(address):
    mongo = pymongo.MongoClient(host=address)
    mongo_db = mongo['nta_runs']
    mongo.nta_runs.Collection.create_index([("date", pymongo.DESCENDING)], expireAfterSeconds=86400)
    # ALL entries into mongo.nta_runs must have datetime.utcnow() timestamp, which is used to delete the record after 86400
    # seconds, 24 hours.
    return mongo_db


def connect_to_mongo_gridfs(address):
    db = pymongo.MongoClient(host=address).nta_storage
    print("Connecting to mongodb at {}".format(address))
    fs = gridfs.GridFS(db)
    return fs

# function to remove columns from a given dataframe, df_in. The columns to be removed are determined by the
# a given list of strings.
def remove_columns(df_in, list_of_columns2remove):
    df = df_in.copy()
    df.drop(list_of_columns2remove, axis=1, inplace=True)
    return df

def reduced_file(df_in):
    df = df_in.copy()
    headers = task_fun.parse_headers(df)
    keeps_str = ['MB_', 'blank', 'blanks', 'BLANK', 'Blank', 'Mean', 'Sub']
    to_drop = [item for sublist in headers for item in sublist if
                        (len(sublist) > 1) & (not any(x in item for x in keeps_str))]
    to_drop.extend(df.columns[(df.columns.str.contains(pat ='CV_|N_Abun_|Median_|STD_')==True)].tolist())
    to_drop.extend(df.columns[(df.columns.str.contains(pat ='Mean_') == True) &
                              (df.columns.str.contains(pat ='MB|blank|blanks|BLANK|Blank|Sub')==False)].tolist())
    if 'Mean_ALLMB' in df.columns.values.tolist():
        to_drop.extend(['Mean_ALLMB'])
    df.drop(to_drop, axis=1, inplace=True)
    return df

def response_log_wrapper(api_name:str):
    def api_log_decorator(request_func):
        def wrapper(*args, **kwargs):
            logger.info(f"============ calling REST API: {api_name}" )
            start_time = time.perf_counter()
            response = request_func(*args, **kwargs)
            logger.info(f"Response: {response}   Run time: {time.perf_counter() - start_time}")
            return response
        return wrapper
    return api_log_decorator

@response_log_wrapper('DSSTOX')
def api_search_masses(masses, accuracy, jobid = "00000"):
    input_json = json.dumps({"search_by": "mass", "query": masses, "accuracy": accuracy})  # assumes ppm
    #if "edap-cluster" in DSSTOX_API:
    api_url = '{}/rest/ms1/batch/{}'.format(DSSTOX_API, jobid)
    #else:
    #    api_url = '{}/nta/rest/ms1/batch/{}'.format(DSSTOX_API, jobid)
    logger.info(api_url)
    http_headers = {'Content-Type': 'application/json'}
    return requests.post(api_url, headers=http_headers, data=input_json)

def api_search_masses_batch(masses, accuracy, batchsize = 50, jobid = "00000"):
    n_masses = len(masses)
    logging.info("Sending {} masses in batches of {}".format(n_masses, batchsize))
    i = 0
    while i < n_masses:
        end = i + batchsize-1
        if end > n_masses-1:
            end = n_masses-1
        response = api_search_masses(masses[i:end+1], accuracy, jobid)
        if not response.ok: # check if we got a successful response
            raise requests.exceptions.HTTPError("Unable to access DSSTOX API. Please contact an administrator or try turning the DSSTox search option off.")
        dsstox_search_json = io.StringIO(json.dumps(response.json()['results'])) # can be an empty string if no hits
        if i == 0:
            dsstox_search_df = pd.read_json(dsstox_search_json, orient='split',
                                        dtype={'TOXCAST_NUMBER_OF_ASSAYS/TOTAL': 'object'})
        else:
            new_search_df = pd.read_json(dsstox_search_json, orient='split',
                                        dtype={'TOXCAST_NUMBER_OF_ASSAYS/TOTAL': 'object'})
            dsstox_search_df = pd.concat([dsstox_search_df, new_search_df], ignore_index = True) #Added ignore index, may not be needed 11/2 MWB
        i = i + batchsize
    
    return dsstox_search_df

@response_log_wrapper('DSSTOX')
def api_search_formulas(formulas, jobID = "00000"):
    input_json = json.dumps({"search_by": "formula", "query": formulas})  # assumes ppm
    if "edap-cluster" in DSSTOX_API:
        api_url = '{}/rest/ms1/batch/{}'.format(DSSTOX_API, jobID)
    else:
        api_url = '{}/nta/rest/ms1/batch/{}'.format(DSSTOX_API, jobID)
    logger.info(api_url)
    http_headers = {'Content-Type': 'application/json'}
    return requests.post(api_url, headers=http_headers, data=input_json)

@response_log_wrapper('HCD')
def api_search_hcd(dtxsid_list):
    post_data = {"chemicals":[], "options": {"cts": None, "minSimilarity": 0.95, "analogsSearchType": None}}
    headers = {'content-type': 'application/json'}
    #url = 'https://hazard.sciencedataexperts.com/api/hazard'
    url = 'https://hcd.rtpnc.epa.gov/api/hazard'
    for dtxsid in dtxsid_list: 
        post_data['chemicals'].append({'chemical': {'sid': dtxsid, "checked": True}, "properties": {}})
    return requests.post(url, data=json.dumps(post_data), headers=headers)
            
def batch_search_hcd(dtxsid_list, batchsize = 1000):
    result_dict = {}
    logger.info(f"Search {len(dtxsid_list)} DTXSIDs in HCD")
    for i in range(0, len(dtxsid_list), batchsize):
        logger.info(f"HCD Query: {i//batchsize} of {len(dtxsid_list)//batchsize} batches")
        response = api_search_hcd(dtxsid_list[i:i+batchsize])
        chem_data_list = json.loads(response.content)['hazardChemicals']
        for chemical in chem_data_list:
            chemical_id = chemical['chemicalId'].split('|')[0]
            result_dict[chemical_id] = {}
            for data in chemical['scores']:
                result_dict[chemical_id][f'{data["hazardName"]}_score'] = data['finalScore']
                result_dict[chemical_id][f'{data["hazardName"]}_authority'] = data['finalAuthority'] if 'finalAuthority' in data.keys() else ''
    return pd.DataFrame(result_dict).transpose().reset_index().rename(columns = {'index':'DTXSID'})

def format_tracer_file(df_in):
    df = df_in.copy()
    # NTAW-94 comment out the following line. Compound is no longer being used
    # df = df.drop(columns=['Compound', 'Score'])
    rt_diff = df['Observed_Retention_Time'] - df['Retention_Time']
    mass_diff = ((df['Observed_Mass'] - df['Monoisotopic_Mass']) / df['Monoisotopic_Mass']) * 1000000
    df.insert(7, 'Mass_Error_PPM', mass_diff)
    df.insert(9, 'Retention_Time_Difference', rt_diff)
    return df

def create_tracer_plot(df_in):
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
    headers = task_fun.parse_headers(df_in)
    abundance = [item for sublist in headers for item in sublist if len(sublist) > 1]
    fig, ax = plt.subplots()
    for i, tracer in df_in.iterrows():
        y = tracer[abundance]
        x = abundance
        ax.plot(x, y, marker='o',label=tracer[0])
        ax.set_ylabel('Log abundance')
        ax.set_xlabel('Sample name')
    #plt.title('Tracers {} mode')
    plt.yscale('log')
    plt.xticks(rotation=-90)
    plt.legend()
    plt.tight_layout()
    sf = ScalarFormatter()
    sf.set_scientific(False)
    ax.yaxis.set_major_formatter(sf)
    ax.margins(x=0.3)
    buffer = io.BytesIO()
    plt.savefig(buffer)#, format='png')
    #plt.show()
    plt.close()
    return buffer.getvalue()

