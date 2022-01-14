
from __future__ import absolute_import
import pandas as pd
import numpy as np
import io
import csv

def parse_mgf(file_in):
    #NFile = file.rsplit('/', 1)[-1]
    #NewFile = NFile.rsplit('.', 1)[0] + ".csv"
    #with open(file) as f:
    decoded_file = file_in.read().decode('utf-8')
    io_string = io.StringIO(decoded_file)
    with io_string as f:
    #print(file)
        RESULT = list()
        for line in f:
            if line.startswith('TITLE'):
                fields = line.split(' ')
                title, MS, of, pmass, charge, at, RT, mins, delimeter = fields
                result = {'MASS': float(pmass), 'RT': float(RT), 'CHARGE': charge, 'FRAG_MASS':[], 'FRAG_INTENSITY':[]}
            if line.startswith('RTINSECONDS'):
                RTS = line.split('=')[1]
                for line in f:
                    if line.split(' ')[0] == 'END':
                        break
                    a, b = line.split('\t')
                    result['FRAG_MASS'].append(float(a))
                    result['FRAG_INTENSITY'].append(float(b))
                RESULT.append(result)
    return RESULT