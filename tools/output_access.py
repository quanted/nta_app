import pandas as pd
import json
import os
from datetime import datetime
from io import StringIO, BytesIO
from zipfile import ZipFile
from django.http import HttpResponse, JsonResponse
from ..app.utilities import connect_to_mongoDB
from ..app.nta_task import FILENAMES
from pymongo.errors import OperationFailure



def datetime_handler(x):
    if isinstance(x, datetime):
        return x.isoformat()
    raise TypeError("Unknown type")


class OutputServer:
    '''
    This class connects to mongodb and servers files from a give nta_run based on its jobID. It returns the files
    in the form of an HttpResponse (content type of either csv data or zip data), ready to be served up by the API.
    '''

    def __init__(self,jobid = '00000000', project_name = None):
        self.jobid = jobid
        self.project_name = ''
        in_docker = os.environ.get("IN_DOCKER")
        self.in_docker = in_docker
        self.mongo = connect_to_mongoDB(in_docker = self.in_docker)
        self.posts = self.mongo.posts
        self.names_toxpi = FILENAMES['toxpi']
        self.names_stats = FILENAMES['stats']
        self.names_tracers = FILENAMES['tracers']
        self.names_cleaned = FILENAMES['cleaned']
        self.names_flags = FILENAMES['flags']
        self.names_combined = FILENAMES['combined']
        self.names_mpp_ready = FILENAMES['mpp_ready']
        self.names_dashboard = FILENAMES['dashboard']
        self.main_file_names = self.names_stats + self.names_cleaned + self.names_flags + [self.names_combined] + \
                               [self.names_mpp_ready] + [self.names_dashboard] + [self.names_toxpi]


    def status(self):
        try:
            id = self.jobid + "_status"
            db_record = self.posts.find_one({'_id': id})
            status = db_record['status']
            time = db_record['date']
            #status = json.dumps(db_record['status'])
            #time = json.dumps(db_record['date'], default = datetime_handler)
        except TypeError:
            status = "Not found"
            time = "Not found"
        response_data = {'start_time': time, 'status': status}
        return JsonResponse(response_data)

    def final_result(self):
        id = self.jobid + "_" + self.names_toxpi
        db_record = self.posts.find_one({'_id': id})
        json_string = json.dumps(db_record['data'])
        df = pd.read_json(json_string, orient='split')
        project_name = db_record['project_name']
        if project_name:
            filename = project_name.replace(" ", "_") + '_' + self.names_toxpi + '.csv'
        else:
            filename = id + '.csv'
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename='+ filename
        df.to_csv(path_or_buf=response, index = False)
        return response

    def all_files(self):
        in_memory_zip = BytesIO()
        #zip = ZipFile(in_memory_zip, 'w')
        with ZipFile(in_memory_zip, 'w') as zip:
            for name in self.main_file_names:
                try:
                    id = self.jobid + "_" + name
                    db_record = self.posts.find_one({'_id': id})
                    json_string = json.dumps(db_record['data'])
                    df = pd.read_json(json_string, orient='split')
                    project_name = db_record['project_name']
                    if project_name:
                        filename = project_name.replace(" ", "_") + '_' + name + '.csv'
                    else:
                        filename = id + '.csv'
                    #csv_string = StringIO()
                    csv_string = df.to_csv(index = False)
                    zip.writestr(filename, csv_string)

                except (OperationFailure, TypeError):
                    return None

                #now do the (optional) tracers file
            for name in self.names_tracers:
                try:
                    id = self.jobid + "_" + name
                    db_record = self.posts.find_one({'_id': id})
                    json_string = json.dumps(db_record['data'])
                    df = pd.read_json(json_string, orient='split')
                    project_name = db_record['project_name']
                    if project_name:
                        filename = project_name.replace(" ", "_") + '_' + name + '.csv'
                    else:
                        filename = id + '.csv'
                    # csv_string = StringIO()
                    csv_string = df.to_csv(index=False)
                    zip.writestr(filename, csv_string)
                except (OperationFailure, TypeError):
                    break


        zip_filename = 'nta_results_' + self.jobid + '.zip'
        response = HttpResponse(in_memory_zip.getvalue(),content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename=' + zip_filename
        response['Content-length'] = in_memory_zip.tell()
        return response




