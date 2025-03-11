import pymongo as pymongo
import gridfs
import os
import logging
import json
import requests

logger = logging.getLogger("nta_app.merge")


def connect_to_mongoDB(address):
    mongo = pymongo.MongoClient(host=address)
    mongo_db = mongo["nta_runs"]
    mongo.nta_runs.Collection.create_index([("date", pymongo.DESCENDING)], expireAfterSeconds=86400)
    # ALL entries into mongo.nta_runs must have datetime.utcnow() timestamp, which is used to delete the record after 86400
    # seconds, 24 hours.
    return mongo_db


def connect_to_mongo_gridfs(address):
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
