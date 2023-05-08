import pandas as pd
import os
import time
from datetime import datetime
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from django.http import HttpResponse, JsonResponse
from ...app.ms1.utilities import connect_to_mongoDB, connect_to_mongo_gridfs
from pymongo.errors import OperationFailure
from gridfs.errors import NoFile

# used for parallelization
# import concurrent.futures
# import collections

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


    #-----------------------------------------------------------------
    #
    # BEGIN PARALLELIZATION OF XLSX GENERATION
    #
    #-----------------------------------------------------------------

    # def generate_excel_sheet(self, name):
    #     try:
    #         print(f'Constructing {name} file')
    #         start = time.perf_counter()
    #         id = self.jobid + "_" + name
    #         db_record = self.gridfs.get(id)
    #         json_string = db_record.read().decode('utf-8')
    #         df = pd.read_json(json_string, orient='split')
    #         return name, df
    #     except Exception as e:
    #         print(e)

    # def generate_excel(self):
    #     file_names = self.gridfs.get(f'{self.jobid}_file_names').read().decode('utf-8').split("&&")
    #     print(f'file_names: {file_names}')

    #     in_memory_buffer = BytesIO()
    #     sheet_dict = collections.OrderedDict()

    #     with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    #         futures = [executor.submit(self.generate_excel_sheet, name) for name in file_names if 'final_' not in name]

    #         for future in concurrent.futures.as_completed(futures):
    #             try:
    #                 name, df = future.result()
    #                 sheet_dict[name] = df
    #             except Exception as e:
    #                 print(e)

    #     with pd.ExcelWriter(in_memory_buffer, engine='openpyxl') as writer:
    #         # for name, df in sheet_dict.items():
    #         #     print(f'name of sheet {name}')
    #         #     df.to_excel(writer, sheet_name=name, index=False)
    #         for name in file_names:
    #             if 'final_' not in name:
    #                 df = sheet_dict.get(name)
    #                 if df is not None:
    #                     print(f'name of sheet {name}')
    #                     df.to_excel(writer, sheet_name=name, index=False)

    #     return in_memory_buffer.getvalue()

    #-----------------------------------------------------------------
    #
    # END PARALLELIZATION OF XLSX GENERATION
    #
    #-----------------------------------------------------------------


    def construct_excel_sheet(self, writer, name):
        try:
            print(f'Constructing {name} file')
            start = time.perf_counter()
            id = self.jobid + "_" + name
            db_record = self.gridfs.get(id)
            json_string = db_record.read().decode('utf-8')
            df = pd.read_json(json_string, orient='split')
            df.to_excel(writer, sheet_name=name, index=False)
            stop = time.perf_counter()
            print(f'Time to construct {name}: {stop - start}')
        except Exception as e:
            print(e)

    def generate_excel(self):
        file_names = self.gridfs.get(f'{self.jobid}_file_names').read().decode('utf-8').split("&&")
        print(f'file_names: {file_names}')
        in_memory_buffer = BytesIO()
        with pd.ExcelWriter(in_memory_buffer, engine='openpyxl') as writer:
            for name in file_names:
                if 'final_' in name:
                    continue
                self.construct_excel_sheet(writer, name)
        return in_memory_buffer.getvalue()
    


    def add_tracer_plots_to_zip(self, zipf,jobid):
        tracer_plots = self.gridfs.get(f'{self.jobid}_tracer_files').read().decode('utf-8').split("&&")
        for name in tracer_plots:
            try:
                tracer_id = jobid + "_" + name
                db_record = self.gridfs.get(tracer_id)
                buffer = db_record.read()
                project_name = db_record.project_name
                if project_name:
                    filename = project_name.replace(" ", "_") + '_' + name + '.png'
                else:
                    filename = tracer_id + '.png'
                zipf.writestr(filename, buffer)
            except (OperationFailure, TypeError, NoFile) as e:
                break

    def final_result(self):
        in_memory_zip = BytesIO()
            
        with ZipFile(in_memory_zip, 'w', ZIP_DEFLATED) as zipf:
            excel_data = self.generate_excel()
            zipf.writestr('summary.xlsx', excel_data)

            self.add_tracer_plots_to_zip(zipf, self.jobid)

        zip_filename = 'nta_results_' + self.jobid + '.zip'
        response = HttpResponse(in_memory_zip.getvalue(),content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename=' + zip_filename
        response['Content-length'] = in_memory_zip.tell()
        return response








