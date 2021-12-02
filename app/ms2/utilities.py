
import pymongo as pymongo
import gridfs
import os
import logging
import json
import requests

logger = logging.getLogger("nta_app.ms2")
logger.setLevel(logging.INFO)

DSSTOX_API = os.environ.get('UBERTOOL_REST_SERVER')
DSSTOX_API = 'http://127.0.0.1:7777'


def connect_to_mongoDB(address):
    mongo = pymongo.MongoClient(host=address)
    mongo_db = mongo['nta_ms2_runs']
    mongo.nta_ms2_runs.Collection.create_index([("date", pymongo.DESCENDING)], expireAfterSeconds=86400)
    # ALL entries into mongo.nta_runs must have datetime.utcnow() timestamp, which is used to delete the record after 86400
    # seconds, 24 hours.
    return mongo_db


def connect_to_mongo_gridfs(address):
    db = pymongo.MongoClient(host=address).nta_ms2_storage
    print("Connecting to mongodb at {}".format(address))
    fs = gridfs.GridFS(db)
    return fs

def ms2_search_api(mass=None, accuracy=None, mode=None, jobid='00000'):
    input_json = json.dumps({"mass": mass, "accuracy": accuracy, "mode": mode})  # assumes ppm
    logger.info("=========== calling MS2 CFMID REST API")
    if "edap-cluster" in DSSTOX_API:
        api_url = '{}/rest/ms2/{}'.format(DSSTOX_API, jobid)
    else:
        api_url = '{}/nta/rest/ms2/{}'.format(DSSTOX_API, jobid)
    logger.info(api_url)
    http_headers = {'Content-Type': 'application/json'}
    response = requests.post(api_url, headers=http_headers, data=input_json)
    cfmid_search_json = io.StringIO(json.dumps(response.json()['results']))
    cfmid_search_df = pd.read_json(dsstox_search_json, orient='split')
    return cfmid_search_df
    