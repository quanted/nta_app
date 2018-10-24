
import pymongo as pymongo
import gridfs
import os

IN_DOCKER = os.environ.get("IN_DOCKER")
rest_url = os.environ['UBERTOOL_REST_SERVER'][:-5] #get address of flask backend for monggodb, no port for flask

#IN_DOCKER = "False"

def connect_to_mongoDB():
    if IN_DOCKER == "False":
        # Dev env mongoDB
        mongo = pymongo.MongoClient(host='mongodb://localhost:27017/0')
        print("MONGODB: mongodb://localhost:27017/0")
    else:
        # Production env mongoDB
        mongo = pymongo.MongoClient(host='mongodb://mongodb:27017/0')
        print("MONGODB: mongodb://mongodb:27017/0")

    mongo_db = mongo['nta_runs']
    mongo.nta_runs.Collection.create_index([("date", pymongo.DESCENDING)], expireAfterSeconds=86400)
    # ALL entries into mongo.flask_hms must have datetime.utcnow() timestamp, which is used to delete the record after 86400
    # seconds, 24 hours.
    return mongo_db


def connect_to_mongo_gridfs():
    if IN_DOCKER == "False":
        # Dev env mongoDB
        db = pymongo.MongoClient(host='mongodb://localhost:27017/0').nta_storage
        print("MONGODB: mongodb://localhost:27017/0")
    else:
        # Production env mongoDB
        db = pymongo.MongoClient(host='mongodb://mongodb:27017/0').nta_storage
        print("MONGODB: mongodb://mongodb:27017/0")

    #mongo_db = mongo['nta_runs']
    fs = gridfs.GridFS(db)
    #db.nta_runs.Collection.create_index([("date", pymongo.DESCENDING)], expireAfterSeconds=86400)
    # ALL entries into mongo.flask_hms must have datetime.utcnow() timestamp, which is used to delete the record after 86400
    # seconds, 24 hours.
    return fs