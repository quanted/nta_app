import aiohttp
import asyncio
import pymongo as pymongo
import gridfs
import os
import logging
import json
import requests
import io
import time
import pandas as pd
from ..feature.feature import MS2_Spectrum

logger = logging.getLogger("nta_app.ms2")
logger.setLevel(logging.INFO)

CHUNK_SIZE = 1000
DSSTOX_API = os.environ.get('UBERTOOL_REST_SERVER')

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

def fetch_ms2_files(jobID):
    gridfs = connect_to_mongo_gridfs(os.environ.get('MONGO_SERVER'))
    file_ids = gridfs.find({'jobid': jobID, 'ms': 'ms2'}).distinct('_id')
    return [(gridfs.get(_id), gridfs.get(_id).filetype)  for _id in file_ids]
        

async def ms2_api_search(output, feature_list, accuracy=None, jobid='00000'):
    """
    Parameters
    ----------
    feature_list : float, optional
        DESCRIPTION. The default is None.
    accuracy : TYPE, optional
        DESCRIPTION. The default is None.
    jobid : TYPE, optional
        DESCRIPTION. The default is '00000'.

    Returns
    -------
    dict: structured as follows:

    {'mass': float,
     'mode': str,
     'data': 
        {(DTXCID, FORMULA, MASS): 
             {'ENERGY1': {'FRAG_MASS': [], 'FRAG_INTENSITY': []},
              'ENERGY2':{'FRAG_MASS': [], 'FRAG_INTENSITY': [])},
              'ENERGY3':{'FRAG_MASS': [], 'FRAG_INTENSITY': []}
             },
        (DTXCID, FORMULA, MASS): 
             {'ENERGY1': {'FRAG_MASS': [], 'FRAG_INTENSITY': []},
              'ENERGY2':{'FRAG_MASS': [], 'FRAG_INTENSITY': [])},
              'ENERGY3':{'FRAG_MASS': [], 'FRAG_INTENSITY': []}
             }, ...
        }
    }
    """
    api_url = validate_url('{}/rest/ms2/{}'.format("https://qed-dev.edap-cluster.com/nta/flask_proxy", jobid))

    async def get_responses(output, feature_list, accuracy, jobid):
        async with aiohttp.ClientSession() as session:
            tasks = []
            for (mass, mode) in feature_list:
                tasks.append(ms2_post(session, mass, mode, 10, api_url, output))
            await asyncio.gather(*tasks)
            
    await get_responses(output, feature_list, accuracy, jobid)

async def ms2_post(session, mass, mode, accuracy, url, output):
    input_json = json.dumps({"mass": mass, "accuracy": accuracy, "mode": mode})
    http_headers = {'Content-Type': 'application/json'}
    async with session.post(url, data = input_json, headers = http_headers) as response:
        data = await response.json()
        output.append(process_response(data, mass, mode))
                
def process_response(*args):
    return cfmid_response_to_spectra(cfmid_response_to_dict(*args))

def cfmid_response_to_dict(response, mass, mode):
    if response['results'] == "none":
        return {'data': None, 'mass': mass, 'mode': mode}
    cfmid_search_json = io.StringIO(json.dumps(response['results']))
    cfmid_search_df = pd.read_json(cfmid_search_json, orient='split')
    cfmid_search_df.rename(columns = {'PMASS_x':'FRAG_MASS', 'INTENSITY0C':'FRAG_INTENSITY'}, inplace = True)
    formated_data_dict = cfmid_search_df.groupby(['DTXCID', 'FORMULA', 'MASS'])[['ENERGY','FRAG_MASS','FRAG_INTENSITY']]\
        .apply(lambda x: x.groupby('ENERGY')[['FRAG_MASS','FRAG_INTENSITY']]\
              .apply(lambda x: x.to_dict('list'))).to_dict('index')
    return {'data': formated_data_dict, 'mass':mass, 'mode':mode}

def cfmid_response_to_spectra(cfmid_response_dict):
    if cfmid_response_dict['data']:
        for cfmid_spectra in cfmid_response_dict['data']:
            cfmid_response_dict['data'][cfmid_spectra]['energy0']= MS2_Spectrum(cfmid_response_dict['data'][cfmid_spectra]['energy0']['FRAG_MASS'], 
                                                                  cfmid_response_dict['data'][cfmid_spectra]['energy0']['FRAG_INTENSITY'])
            cfmid_response_dict['data'][cfmid_spectra]['energy1'] = MS2_Spectrum(cfmid_response_dict['data'][cfmid_spectra]['energy1']['FRAG_MASS'], 
                                                                  cfmid_response_dict['data'][cfmid_spectra]['energy1']['FRAG_INTENSITY'])
            cfmid_response_dict['data'][cfmid_spectra]['energy2'] = MS2_Spectrum(cfmid_response_dict['data'][cfmid_spectra]['energy2']['FRAG_MASS'], 
                                                                  cfmid_response_dict['data'][cfmid_spectra]['energy2']['FRAG_INTENSITY'])
    return cfmid_response_dict

def validate_url(url):
    if 'https://' not in url:
        return f'https://{url}'
    return url
