import pandas as pd
import json
import os
import openpyxl
import time
from datetime import datetime
from io import StringIO, BytesIO
from PIL import Image
from zipfile import ZipFile, ZIP_DEFLATED
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from ...app.merge.utilities import connect_to_mongoDB, connect_to_mongo_gridfs
from pymongo.errors import OperationFailure
from gridfs.errors import NoFile

#os.environ['IN_DOCKER'] = "False" #for local dev - also see similar switch in app/nta_task.py

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
        self.in_docker = os.environ.get("IN_DOCKER") != "False"
        self.mongo_address = os.environ.get('MONGO_SERVER')
        self.mongo = connect_to_mongoDB(self.mongo_address)
        self.gridfs = connect_to_mongo_gridfs(self.mongo_address)
        self.posts = self.mongo.posts

    def status(self):
        try:
            search_id = self.jobid + "_status"
            db_record = self.posts.find_one({'_id': search_id})
            status = db_record['status']
            time = db_record['date']
            except_text = db_record['error_info']
        except TypeError:
            status = "Not found"
            time = "Not found"
            except_text = "Not found"
        response_data = {'start_time': time, 'status': status, 'error_info': except_text}
        return JsonResponse(response_data)

    def final_result(self):
        self.file_names = self.gridfs.get(f'{self.jobid}_file_names').read().decode('utf-8').split("&&")
        if len(self.file_names) == 1:
            return self._package_csv()
        return self._package_xlsx()

    def _package_xlsx(self):
        print(f'Serving xlsx file')
        initial = time.perf_counter()
        in_memory_xlsx = BytesIO()
        with pd.ExcelWriter(in_memory_xlsx, engine = 'openpyxl') as writer:
            for file in self.file_names:
                try:                                             
                    start = time.perf_counter()
                    id = self.jobid + "_" + file
                    db_record = self.gridfs.get(id)
                    json_string = db_record.read().decode('utf-8')
                    df = pd.read_json(json_string, orient='split')
                    df.to_excel(writer, sheet_name=file, index = False)
                    stop = time.perf_counter()
                    print(f'========= Time to construct sheet {file}: {stop - start}')
                except Exception as e:
                    print(e)
                    continue
        response = StreamingHttpResponse(in_memory_xlsx.getvalue(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = f'attachment; filename=merged_summary_{self.jobid}.xlsx'
        end = time.perf_counter()
        print(f'Time to get xlsx: {end - initial}')
        return response

    def _package_csv(self):
        in_memory_zip = BytesIO()
        with ZipFile(in_memory_zip, 'w', ZIP_DEFLATED) as zipf:
            for file in self.file_names:
                try:
                    record_id = self.jobid + "_" + file
                    db_record = self.gridfs.get(record_id)
                    json_string = db_record.read().decode('utf-8')
                    df = pd.read_json(json_string, orient='split')
                    project_name = db_record.project_name
                    if project_name:
                        filename = project_name.replace(" ", "_") + '_' + file + '.csv'
                    else:
                        filename = record_id + '.csv'
                    csv_string = df.to_csv(index = False)
                    zipf.writestr(filename, csv_string)
                except (OperationFailure, TypeError, NoFile) as e:
                    break
        zip_filename = 'nta_results_merged' + self.jobid + '.zip'
        response = HttpResponse(in_memory_zip.getvalue(),content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename=' + zip_filename
        response['Content-length'] = in_memory_zip.tell()
        return response
        