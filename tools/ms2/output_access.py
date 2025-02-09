import pandas as pd
import json
import os
from datetime import datetime
from io import StringIO, BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from django.http import HttpResponse, JsonResponse
from ...app.ms2.utilities import connect_to_mongoDB, connect_to_mongo_gridfs
from ...app.ms2.ms2_task import FILENAMES
from pymongo.errors import OperationFailure
from gridfs.errors import NoFile

# os.environ['IN_DOCKER'] = "False" #for local dev - also see similar switch in app/nta_task.py


def datetime_handler(x):
    if isinstance(x, datetime):
        return x.isoformat()
    raise TypeError("Unknown type")


class OutputServer:
    """
    This class connects to mongodb and servers files from a give nta_run based on its jobID. It returns the files
    in the form of an HttpResponse (content type of either csv data or zip data), ready to be served up by the API.
    """

    def __init__(self, jobid="00000000", project_name=None):
        self.jobid = jobid
        self.project_name = ""
        self.in_docker = os.environ.get("IN_DOCKER") != "False"
        self.mongo_address = os.environ.get("MONGO_SERVER")
        self.mongo = connect_to_mongoDB(self.mongo_address)
        self.gridfs = connect_to_mongo_gridfs(self.mongo_address)
        self.posts = self.mongo.posts
        # self.names_duplicates = FILENAMES['duplicates']
        self.names_output = FILENAMES["final_output"]
        self.main_file_names = self.names_output

    def status(self):
        try:
            search_id = self.jobid + "_status"
            db_record = self.posts.find_one({"_id": search_id})
            status = db_record["status"]
            time = db_record["date"]
            except_text = db_record["error_info"]
            n_masses = db_record["n_masses"]
            progress = db_record["progress"]
            # status = json.dumps(db_record['status'])
            # time = json.dumps(db_record['date'], default = datetime_handler)
        except TypeError:
            status = "Not found"
            time = "Not found"
            except_text = "Not found"
            n_masses = "Not found"
            progress = "Not found"
        response_data = {
            "start_time": time,
            "status": status,
            "error_info": except_text,
            "n_masses": n_masses,
            "progress": progress,
        }
        return JsonResponse(response_data)

    # def final_result(self):
    #     id = self.jobid + "_" + self.names_toxpi[1]
    #     #id = self.jobid + "_" + self.names_duplicates[0]# TODO remove
    #     #db_record = self.posts.find_one({'_id': id})
    #     #json_string = json.dumps(db_record['data'])
    #     db_record = self.gridfs.get(id)
    #     json_string = db_record.read().decode('utf-8')
    #     df = pd.read_json(json_string, orient='split')
    #     #project_name = db_record['project_name']
    #     project_name = db_record.project_name
    #     if project_name:
    #         filename = project_name.replace(" ", "_") + '_' + self.names_toxpi[1] + '.csv'
    #         #filename = project_name.replace(" ", "_") + '_' + self.names_duplicates[0] + '.csv' # TODO remove
    #     else:
    #         filename = id + '.csv'
    #     response = HttpResponse(content_type='text/csv')
    #     response['Content-Disposition'] = 'attachment; filename='+ filename
    #     df.to_csv(path_or_buf=response, index = False)
    #     return response

    def all_files(self):
        in_memory_zip = BytesIO()
        # zip = ZipFile(in_memory_zip, 'w')
        with ZipFile(in_memory_zip, "w", ZIP_DEFLATED) as zipf:
            for name in self.main_file_names:
                print(name)
                try:
                    record_id = self.jobid + "_" + name
                    # db_record = self.posts.find_one({'_id': record_id})
                    # json_string = json.dumps(db_record['data'])
                    db_record = self.gridfs.get(record_id)
                    json_string = db_record.read().decode("utf-8")
                    df = pd.read_json(json_string, orient="split")
                    # project_name = db_record['project_name']
                    project_name = db_record.project_name
                    if project_name:
                        filename = project_name.replace(" ", "_") + "_" + name + ".csv"
                    else:
                        filename = record_id + ".csv"
                    # csv_string = StringIO()

                    if (
                        "Q-SCORE" in df.columns
                    ):  # Add check that the file is a results dataframe and not the input parameters
                        df["Q-SCORE"] = pd.to_numeric(df["Q-SCORE"], errors="coerce")
                        df["PERCENTILE"] = pd.to_numeric(df["PERCENTILE"], errors="coerce")
                        df["SUM_SCORE"] = pd.to_numeric(df["SUM_SCORE"], errors="coerce")
                        df["Q-SCORE"] = df["Q-SCORE"].round(2)
                        df["PERCENTILE"] = df["PERCENTILE"].round(2)

                    csv_string = df.to_csv(index=False)
                    zipf.writestr(filename, csv_string)

                except (OperationFailure, TypeError, NoFile) as e:
                    pass

        zip_filename = "nta_ms2_results_" + self.jobid + ".zip"
        response = HttpResponse(in_memory_zip.getvalue(), content_type="application/zip")
        response["Content-Disposition"] = "attachment; filename=" + zip_filename
        response["Content-length"] = in_memory_zip.tell()
        return response
