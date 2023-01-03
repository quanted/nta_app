
from __future__ import absolute_import
import pandas as pd
import numpy as np
import io
import csv
#import pyopenms as pyms

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("nta_app.ms2")
logger.setLevel(logging.INFO)

class MS2_Parser():
    """

    """
    @staticmethod
    def parse_file(file, **kwargs):
        parser = MS2_Parser._get_parser(file.filetype)
        return parser(file, **kwargs)
    
    def _get_parser(file_type):
        if file_type == 'mgf':
            return MS2_Parser._mgf_parser
        elif file_type == 'mzml':
            return MS2_Parser._mzml_parser
        else:
            raise ValueError(file_type)

    def _mgf_parser(file_in):
        """
        Output: list of dictionaries sturcutres as follows:
                [{'Mass': float, 'RT': float, 'CHARGE': str, 'FRAG_MASS': [float], 'FRAG_INTENSITY':[float]}, ...]
                
                Each list entry represents a feature present within the MS2 dataset
        """
        OUTPUT = []
        with Open_Input(file_in) as file:
            for line in file:
                if line.startswith('BEGIN IONS'):
                    result = {'MASS': None, 'RT': None, 'CHARGE': None, 'FRAG_MASS':[], 'FRAG_INTENSITY':[]}
                elif line.startswith('PEPMASS'):
                    result['MASS'] = float(MS2_Parser._seperate_line(line.split('=')[1])[0])
                elif line.startswith('RTINSECONDS'):
                    result['RT'] = float(line.split('=')[1])
                elif line.startswith('CHARGE'):
                    result['CHARGE'] = line.split('=')[1]
                elif line.startswith('TITLE'):
                    continue    #TODO: Add functinality later to include TITLE to track file origin
                elif line.startswith('END IONS'):
                    OUTPUT.append(result)
                else:
                     mass_frag, frag_intensity = MS2_Parser._seperate_line(line)
                     result['FRAG_MASS'].append(float(mass_frag))
                     result['FRAG_INTENSITY'].append(float(frag_intensity))
        return OUTPUT
    
    def _mzml_parser(file_in):
        # with Open_Input(file_in) as datafile:
        #     file = datafile.read()
        #     OUTPUT = []
        #     exp = pyms.MSExperiment()
        #     pyms.MzMLFile().load(file, exp)
        #     for spectrum in exp.spectrum():
        #         mass, intensity = spectrum.get_peaks()
        #         precursor = spectrum.getPrecursors()[0]
        #         OUTPUT.append({'MASS': precursor.get_mz(), 'RT': spectrum.getRT(), 'CHARGE': precursor.getCharge(), 'FRAGMASS': list(mass), 'FRAG_INTENSITY' : list(intensity)})  
        # return OUTPUT
        pass
    
    def _seperate_line(line):
        if " " in line:
            return line.split(" ")
        elif "\t" in line:
            return line.split('\t')
    
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
	        
# def mzml_parser(file_in):
#     with Open_Input(file_in) as file:
#         OUTPUT = []
#         exp = pyms.MSExperiment()
#         pyms.MzMLFile().load(file, exp)
#         for spectrum in exp.spectrum():
#             mass, intensity = spectrum.get_peaks()
#             precursor = spectrum.getPrecursors()[0]
#             OUTPUT.append({'MASS': precursor.get_mz(), 'RT': spectrum.getRT(), 'CHARGE': precursor.getCharge(), 'FRAGMASS': list(mass), 'FRAG_INTENSITY' : list(intensity)})  
#     return OUTPUT

