
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
CCD_API = os.environ.get('CCD_API')
CCD_API_KEY = os.environ.get('CCD_API_KEY')


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

def api_search_mass_batch(mass_list, accuracy):
    api_url = '{}/chemical/msready/search/by-mass/'.format(CCD_API)
    http_headers = {'x-api-key': CCD_API_KEY, 'content-type': 'application/json',
                    'accept': 'application/json'}
    post_data = {"masses": mass_list, "error": accuracy}
    response = requests.post(api_url, data=json.dumps(post_data), headers=http_headers)
    candidate_list = response.json()
    return candidate_list

def api_search_formula(formula):
    api_url = '{}/chemical/msready/search/by-formula/{}'.format(CCD_API, formula)
    http_headers = {'x-api-key': CCD_API_KEY}
    response = requests.get(api_url, headers=http_headers)
    return response.json()

def api_get_metadata(dtxsid):
    http_headers = {'x-api-key': CCD_API_KEY}
    chem_details_api = '{}/chemical/detail/search/by-dtxsid/{}'.format(CCD_API,dtxsid)
    chem_details_response = requests.get(chem_details_api, headers=http_headers)
    return {dtxsid: chem_details_response.json()}

def api_get_metadata_batch(dtxsid_list):
    http_headers = {'x-api-key': CCD_API_KEY, 'content-type': 'application/json',
                    'accept': 'application/json'}
    chem_details_api = '{}/chemical/detail/search/by-dtxsid/?projection=ntatoolkit'.format(CCD_API)
    metadata_response = requests.post(chem_details_api, data=json.dumps(dtxsid_list),
                                          headers=http_headers)
    return metadata_response.json()

@response_log_wrapper('CCD API by mass')
def api_search_mass_list(masses, accuracy, batchsize=100):
    n_masses = len(masses)
    logging.info("Searching {} masses in batches of {} using the CCD API".format(n_masses, batchsize))
    candidates_dict = {}
    for i in range(0, n_masses, batchsize):
        candidates_batch = api_search_mass_batch(mass_list=masses[i:i+batchsize],
                              accuracy=accuracy)
        candidates_dict.update(candidates_batch)
    logger.info("done with mass searching")
    metadata_list = []
    for parent, candidates in candidates_dict.items():
        dtxsid_batch_size = 1000 # loop through 1000 DTXSIDs max
        for i in range(0, len(candidates), dtxsid_batch_size): 
            metadata = api_get_metadata_batch(candidates[i:i+dtxsid_batch_size])
            metadata_df = pd.DataFrame(metadata)
            metadata_df['Input'] = float(parent)
            metadata_list.append(metadata_df)
    logger.info("done with metadata searching")
    results_df = pd.concat(metadata_list)
    return results_df

@response_log_wrapper('CCD API by formula')
def api_search_formula_list(formulas):
    n_formulas = len(formulas)
    logging.info("Searching {} formulas in batches of 1 using the CCD API".format(n_formulas))
    candidates_dict = {}
    for formula in formulas:
        candidates_batch = api_search_formula(formula)
        candidates_dict.update({formula:candidates_batch})
    metadata_list = []
    for parent, candidates in candidates_dict.items():
        dtxsid_batch_size = 1000 # loop through 1000 DTXSIDs max
        for i in range(0, len(candidates), dtxsid_batch_size): 
            metadata = api_get_metadata_batch(candidates[i:i+dtxsid_batch_size])
            metadata_df = pd.DataFrame(metadata)
            metadata_df['Input'] = parent
            metadata_list.append(metadata_df)
    results_df = pd.concat(metadata_list)
    return results_df

def api_search_hcd(dtxsid_list):
    post_data = {"chemicals":[], "options": {"noRecords": "true", "usePredictions": "false"}}
    headers = {'content-type': 'application/json'}
    #url = 'https://hazard.sciencedataexperts.com/api/hazard'
    url = 'https://hcd.rtpnc.epa.gov/api/hazard'
    for dtxsid in dtxsid_list: 
        post_data['chemicals'].append({'chemical': {'sid': dtxsid, "checked": True}, "properties": {}})
    return requests.post(url, data=json.dumps(post_data), headers=headers)
            
@response_log_wrapper('HCD')
def batch_search_hcd(dtxsid_list, batchsize = 200):
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

def metadata_posthoc_formatting(metadata_df):
    metadata_df['mass_difference'] = metadata_df['Input'] - metadata_df['monoisotopicMass']
    #metadata_df - metadata_df.drop(columns=['expocat'])
    metadata_df.rename(columns={"expocatMedianPrediction": "EXPOCAST_MEDIAN_EXPOSURE_PREDICTION_MG/KG-BW/DAY",
                                "expocat":"inExpocast?",
                                "nhanes":"inNHANES?"})
    metadata_df['inExpocast?'] = metadata_df['inExpocast?'].astype(bool)
    metadata_df['inNHANES?'] = metadata_df['inNHANES?'].astype(bool)
    
    
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

