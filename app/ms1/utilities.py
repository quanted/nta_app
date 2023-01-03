
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
from .functions_Universal_v3 import parse_headers

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


def reduced_file(df_in):
    df = df_in.copy()
    headers = parse_headers(df, 0)
    keeps_str = ['MB_', 'blank', 'blanks', 'BLANK', 'Blank', 'Median', 'Sub']
    to_drop = [item for sublist in headers for item in sublist if
                        (len(sublist) > 1) & (not any(x in item for x in keeps_str))]
    to_drop.extend(df.columns[(df.columns.str.contains(pat ='CV_|N_Abun_|Mean_|STD_')==True)].tolist())
    to_drop.extend(df.columns[(df.columns.str.contains(pat ='Median_') == True) &
                              (df.columns.str.contains(pat ='MB|blank|blanks|BLANK|Blank|Sub')==False)].tolist())
    if 'Median_ALLMB' in df.columns.values.tolist():
        to_drop.extend(['Median_ALLMB'])
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
        dsstox_search_json = io.StringIO(json.dumps(response.json()['results']))
        if i == 0:
            dsstox_search_df = pd.read_json(dsstox_search_json, orient='split',
                                        dtype={'TOXCAST_NUMBER_OF_ASSAYS/TOTAL': 'object'})
        else:
            dsstox_search_df = dsstox_search_df.append(pd.read_json(dsstox_search_json, orient='split',
                                        dtype={'TOXCAST_NUMBER_OF_ASSAYS/TOTAL': 'object'}), ignore_index = True) #Added ignore index, may not be needed 11/2 MWB
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
    url = 'https://hazard.sciencedataexperts.com/api/hazard'
    for dtxsid in dtxsid_list: 
        post_data['chemicals'].append({'chemical': {'sid': dtxsid, "checked": True}, "properties": {}})
    return requests.post(url, data=json.dumps(post_data), headers=headers)
            
def batch_search_hcd(dtxsid_list, batchsize = 150):
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
    df = df.drop(columns=['Compound', 'Score'])
    rt_diff = df['Observed_Retention_Time'] - df['Retention_Time']
    mass_diff = ((df['Observed_Mass'] - df['Monoisotopic_Mass']) / df['Monoisotopic_Mass']) * 1000000
    df.insert(7, 'Mass_Error_PPM', mass_diff)
    df.insert(9, 'Retention_Time_Difference', rt_diff)
    return df

def create_tracer_plot(df_in):
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
    headers = parse_headers(df_in, 0)
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

