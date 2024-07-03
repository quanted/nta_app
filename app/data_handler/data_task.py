import os
import logging
import json
import datetime

from .utilities import connect_to_mongo_gridfs, get_mongoDB


NO_DASK = False  # set this to True to run locally without dask (for debug purposes)

logger = logging.getLogger("nta_app.utilities")

MONGO_SERVER = os.environ.get("MONGO_SERVER")

# def store_data(path, input_data):
# to_save = json.dumps(input_data)
# gridfs = connect_to_mongo_gridfs(mongo_address)
#    gridfs.put(to_save, filename ="TEST/PATH1", _id="TEST/PATH1", encoding='utf-8')


def delete_data(filename, jobid, ms):
    gridfs = connect_to_mongo_gridfs(MONGO_SERVER)
    mongoDB = get_mongoDB(MONGO_SERVER)
    files = mongoDB.get_collection("fs.files")
    for ID in files.find({"filename": filename, "jobid": jobid, "ms": ms}).distinct("_id"):
        gridfs.delete(ID)


def get_filenames(jobid, ms):
    resp_dict = {"Neg": [], "Pos": []}
    gridfs = connect_to_mongo_gridfs(MONGO_SERVER)
    mongoDB = get_mongoDB(MONGO_SERVER)
    files = mongoDB.get_collection("fs.files")
    for ID in files.find({"jobid": jobid, "ms": ms, "mode": "neg"}).distinct("_id"):
        resp_dict["Neg"].append(gridfs.get(ID).filename)
    for ID in files.find({"jobid": jobid, "ms": ms, "mode": "pos"}).distinct("_id"):
        resp_dict["Pos"].append(gridfs.get(ID).filename)
    return json.dumps(resp_dict)


def get_grid_db():
    gridfs = connect_to_mongo_gridfs(MONGO_SERVER)
    return gridfs


def handle_uploaded_file(file, filename, filetype, ms, mode, jobid):
    gridfs_df = get_grid_db()
    file_id = gridfs_df.put(
        file,
        filename=filename,
        filetype=filetype,
        encoding="utf-8",
        ms=ms,
        mode=mode,
        jobid=jobid,
    )

    mongoDB = get_mongoDB(MONGO_SERVER)
    files = mongoDB.get_collection("fs.files")
    chunks = mongoDB.get_collection("fs.chunks")

    files.create_index([("uploadDate", 1)], expireAfterSeconds=86400)  # Expires in 24h

    chunks.update_many({"files_id": file_id}, {"$set": {"uploadDate": datetime.datetime.utcnow()}})
    chunks.create_index([("uploadDate", 1)], expireAfterSeconds=86460)  # Expires in 24h

    return file_id


def list(self):
    """List the names of all files stored in this instance of
    :class:`GridFS`.
    .. versionchanged:: 3.1
       ``list`` no longer ensures indexes.
    """
    # With an index, distinct includes documents with no filename
    # as None.
    return [name for name in self.__files.distinct("filename") if name is not None]
