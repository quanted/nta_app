
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
    with Open_Input(file_in) as file:
        result = {}
        OUTPUT = []
        for line in file:
            if line.startswith('BEGIN IONS'):
                result = {'MASS': None, 'RT': None, 'CHARGE': None, 'FRAG_MASS':[], 'FRAG_INTENSITY':[]}
            elif line.startswith('PEPMASS'):
                result['MASS'] = float(line.split('=')[1])
            elif line.startswith('RTINSECONDS'):
                result['RT'] = float(line.split('=')[1])
            elif line.startswith('CHARGE'):
                result['CHARGE'] = line.split('=')[1]
            elif line.startswith('TITLE'):
                continue    #TODO: Add functinality later to include TITLE to track file origin
            elif line.startswith('END IONS'):
                OUTPUT.append(result)
            else:
                 mass_frag , frag_intensity = line.split('\t')
                 result['FRAG_MASS'].append(float(mass_frag))
                 result['FRAG_INTENSITY'].append(float(frag_intensity))
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
	        


