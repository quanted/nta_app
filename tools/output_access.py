import pandas as pd
import numpy as np
import json
import gridfs
import sys
import os
import csv
import time
from datetime import datetime
from django.http import HttpResponse
from ..app.utilities import connect_to_mongoDB


class OutputServer:

    def __init__(self,jobid = '00000000', project_name = None):
        self.jobid = jobid
        self.project_name = ''
        self.mongo = connect_to_mongoDB()
        self.posts = self.mongo.posts
        self.name_toxpi = 'combined_toxpi'

    def final_result(self):
        id = self.jobid + "_" + self.name_toxpi
        db_record = self.posts.find_one({'_id': id})
        json_string = json.dumps(db_record['data'])
        df = pd.read_json(json_string, orient='split')
        project_name = db_record['project_name']
        if project_name:
            filename = project_name.replace(" ", "_") + '_' + self.name_toxpi + '.csv'
        else:
            filename = id + '.csv'
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename='+ filename
        df.to_csv(path_or_buf=response, index = False)
        return response
