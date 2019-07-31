
import pymongo as pymongo
import gridfs
import os
import logging
import requests
from .functions_Universal_v3 import parse_headers

logger = logging.getLogger("nta_app")
logger.setLevel(logging.INFO)

DSSTOX_API = os.environ.get('DSSTOX_API')
DSSTOX_API = '127.0.0.1:5050'

MONGO_ADDRESS = os.environ.get('MONGO_SERVER')

def connect_to_mongoDB():
    mongo = pymongo.MongoClient(host=MONGO_ADDRESS)
    mongo_db = mongo['nta_runs']
    mongo.nta_runs.Collection.create_index([("date", pymongo.DESCENDING)], expireAfterSeconds=86400)
    # ALL entries into mongo.nta_runs must have datetime.utcnow() timestamp, which is used to delete the record after 86400
    # seconds, 24 hours.
    return mongo_db


def connect_to_mongo_gridfs():
    db = pymongo.MongoClient(host=MONGO_ADDRESS).nta_storage
    print(MONGO_ADDRESS)
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


def search_mass(masses, accuracy, units, jobID = "00000"):
    input_json = self.format_varroapop_payload()
    logger.info("=========== calling DSSTOX REST API")
    api_url = '{}/nta/rest/nta/batch/{}/'.format(DSSTOX_API, jobID)
    logger.info(api_url)
    http_headers = {'Content-Type': 'application/json'}
    #logger.info("JSON payload:")
    #print(input_json)
    return requests.post(api_url, headers=http_headers, data=input_json, timeout=60)