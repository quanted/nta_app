
from __future__ import absolute_import
import os
import logging
import json
import uuid
from datetime import datetime

from flask import request, Response
from flask_restful import Resource

import pymongo as pymongo

IN_DOCKER = os.environ.get("IN_DOCKER")
rest_url = os.environ['UBERTOOL_REST_SERVER'] #get address of flask backend for monggodb


def connect_to_mongoDB():
    if IN_DOCKER == "False":
        # Dev env mongoDB
        mongo = pymongo.MongoClient(host='mongodb://localhost:27017/0')
        print("MONGODB: mongodb://localhost:27017/0")
    else:
        # Production env mongoDB
        #mongo = pymongo.MongoClient(host='mongodb://mongodb:27017/0')
        mongo_url = rest_url+':27017/0'
        mongo = pymongo.MongoClient(host=mongo_url)
        print("MONGODB: "+ mongo_url)
    mongo_db = mongo['pram_tasks']
    mongo.pram_tasks.Collection.create_index([("date", pymongo.DESCENDING)], expireAfterSeconds=86400)
    # ALL entries into mongo.flask_hms must have datetime.utcnow() timestamp, which is used to delete the record after 86400
    # seconds, 24 hours.
    return mongo_db