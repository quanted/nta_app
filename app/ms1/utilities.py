
import pymongo as pymongo
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import gridfs
import io
import os
import logging
import json
import requests
import numpy as np
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


def api_search_masses(masses, accuracy, jobID = "00000"):
    print("Sending {} masses".format(len(masses)))
    input_json = json.dumps({"search_by": "mass", "query": masses, "accuracy": accuracy})  # assumes ppm
    logger.info("=========== calling DSSTOX REST API")
    api_url = '{}/nta/rest/ms1/batch/{}'.format(DSSTOX_API, jobID)
    logger.info(api_url)
    http_headers = {'Content-Type': 'application/json'}
    return requests.post(api_url, headers=http_headers, data=input_json)



def api_search_formulas(formulas, jobID = "00000"):
    input_json = json.dumps({"search_by": "formula", "query": formulas})  # assumes ppm
    logger.info("=========== calling DSSTOX REST API")
    api_url = '{}/nta/rest/ms1/batch/{}'.format(DSSTOX_API, jobID)
    logger.info(api_url)
    http_headers = {'Content-Type': 'application/json'}
    return requests.post(api_url, headers=http_headers, data=input_json)


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

