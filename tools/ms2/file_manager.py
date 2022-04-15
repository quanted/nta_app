
from __future__ import absolute_import
import pandas as pd
import numpy as np
import io
import csv

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("nta_app.ms2")
logger.setLevel(logging.INFO)

def parse_mgf(file_in):
    f = file_in.read().decode('utf-8')
    OUTPUT = list()
    filtered_text = f.replace(" \r", "").replace("\r", "").replace('END IONS', "")
    formated_list = list(map(lambda x: list(filter(str.strip, x.split('\n'))), filtered_text.split('BEGIN IONS')))
    formated_list = list(filter(None, formated_list))
    for data_block in formated_list:
        result = {'MASS': None, 'RT': None, 'CHARGE': None, 'FRAG_MASS':[], 'FRAG_INTENSITY':[]}
        for entry in data_block:
            if entry.startswith('PEPMASS'):
                result['MASS'] = float(entry.split('=')[1])
            elif entry.startswith('RTINSECONDS'):
                result['RT'] = float(entry.split('=')[1])
            elif entry.startswith('CHARGE'):
                result['CHARGE'] = entry.split('=')[1]
            elif entry.startswith('TITLE'):
                continue    #TODO: Add functinality later to include TITLE to track file origin
            elif '\t' in entry:
                 mass_frag , frag_intensity = entry.split('\t')
                 result['FRAG_MASS'].append(float(mass_frag))
                 result['FRAG_INTENSITY'].append(float(frag_intensity))
        OUTPUT.append(result)
    return OUTPUT

class Open_Input(object):
    def __init__(self, file_in):
        if isinstance(file_in, str):
            self.file_obj = open(file_in, "r")
        else:
            decoded_file = file_in.read().decode('utf-8')
            self.file_obj = io.StringIO(decoded_file)
    def __enter__(self):
    	return self.file_obj
    def __exit__(self, type, value, traceback):
        self.file_obj.close()
        return True
	        


