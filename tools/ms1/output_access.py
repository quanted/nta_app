import pandas as pd
import json
import os
import openpyxl
import time
from datetime import datetime
from io import StringIO, BytesIO
from PIL import Image
from zipfile import ZipFile, ZIP_DEFLATED
from django.http import HttpResponse, JsonResponse
from ...app.ms1.utilities import connect_to_mongoDB, connect_to_mongo_gridfs
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
        #
        #This code and the output code need to be optimized to improve compute time to generete xlsx file and transfer time of file to client
        #
        initial = time.perf_counter()
        in_memory_buffer = BytesIO()
        file_names = self.gridfs.get(f'{self.jobid}_file_names').read().decode('utf-8').split("&&")
        with pd.ExcelWriter(in_memory_buffer, engine='openpyxl') as writer:
            for name in file_names:
                if 'final_' in name:          #Added to skip including the 'full_ouput' data in the final results xlsx. Doubles file size and greatly increases
                    continue                        #compute time to preapre the file and transfer time to get file to client. Need better solution down road (compression)
                try:        
                    print(f'Constructing {name} file')                                                
                    start = time.perf_counter()
                    id = self.jobid + "_" + name
                    db_record = self.gridfs.get(id)
                    json_string = db_record.read().decode('utf-8')
                    df = pd.read_json(json_string, orient='split')
                    df.to_excel(writer, sheet_name=name, index = False)
                    stop = time.perf_counter()
                    print(f'Time to construct {name}: {stop - start}')
                except Exception as e:
                    print(e)
                    continue
        response = HttpResponse(in_memory_buffer.getvalue(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = 'attachment; filename=summary.xlsx'
        end = time.perf_counter()
        print(f'Time to get xlsx: {end - initial}')
        return response

    def all_files(self):
        in_memory_zip = BytesIO()
        file_names = self.gridfs.get(f'{self.jobid}_file_names').read().decode('utf-8').split("&&")
        tracer_plots = self.gridfs.get(f'{self.jobid}_tracer_files').read().decode('utf-8').split("&&")
        
        with ZipFile(in_memory_zip, 'w', ZIP_DEFLATED) as zipf:
            for name in file_names:
                try:
                    record_id = self.jobid + "_" + name
                    db_record = self.gridfs.get(record_id)
                    json_string = db_record.read().decode('utf-8')
                    df = pd.read_json(json_string, orient='split')
                    project_name = db_record.project_name
                    if project_name:
                        filename = project_name.replace(" ", "_") + '_' + name + '.csv'
                    else:
                        filename = record_id + '.csv'
                    csv_string = df.to_csv(index = False)
                    zipf.writestr(filename, csv_string)

                except (OperationFailure, TypeError, NoFile) as e:
                    break

            for name in tracer_plots:
                try:
                    tracer_id = self.jobid + "_" + name
                    db_record = self.gridfs.get(tracer_id)
                    buffer = db_record.read()
                    project_name = db_record.project_name
                    if project_name:
                        filename = project_name.replace(" ", "_") + '_' + name + '.png'
                    else:
                        filename = tracer_id + '.png'
                    csv_string = df.to_csv(index=False)
                    zipf.writestr(filename, buffer)
                except (OperationFailure, TypeError, NoFile) as e:
                    break


        zip_filename = 'nta_results_' + self.jobid + '.zip'
        response = HttpResponse(in_memory_zip.getvalue(),content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename=' + zip_filename
        response['Content-length'] = in_memory_zip.tell()
        return response




