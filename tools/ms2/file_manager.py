from __future__ import absolute_import
import pandas as pd
import numpy as np
import io
import csv

# import pyopenms as pyms

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("nta_app.ms2")
logger.setLevel(logging.INFO)


class MS2_Parser:
    """ """

    @staticmethod
    def parse_file(file, **kwargs):
        parser = MS2_Parser._get_parser(file.filetype)
        return parser(file, **kwargs)

    def _get_parser(file_type):
        if str(file_type).lower() == "mgf":
            return MS2_Parser._mgf_parser
        elif str(file_type).lower() == "msp":
            return MS2_Parser._msp_parser
        elif str(file_type).lower() == "mzml":
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
            all_lines = file.readlines()
            non_blank_lines = [
                line for line in all_lines if line.strip()
            ]  # Remove empty lines from the input text lines

            for line in non_blank_lines:
                line = line.strip()  # Get rid of potential blank spaces at end of line
                if line.startswith("BEGIN IONS"):
                    result = {
                        "MASS": None,
                        "RT": None,
                        "CHARGE": None,
                        "FRAG_MASS": [],
                        "FRAG_INTENSITY": [],
                        "filename": file.filename,
                    }
                elif line.startswith("PEPMASS"):
                    line = line.split(" ")[
                        0
                    ]  # Get rid of extra intensity value that is present in Thermo data after a space
                    # result['MASS'] = float(MS2_Parser._seperate_line(line.split('=')[1])[0])
                    result["MASS"] = float(line.split("=")[1])
                elif line.startswith("RTINSECONDS"):
                    result["RT"] = float(line.split("=")[1])
                elif line.startswith("CHARGE"):
                    result["CHARGE"] = line.split("=")[1]
                elif line.startswith("TITLE"):
                    continue  # TODO: Add functinality later to include TITLE to track file origin
                elif line.startswith("END IONS"):
                    OUTPUT.append(result)
                elif line[0].isdigit():  # Only grab fragment data if the line starts with a numeric value
                    mass_frag, frag_intensity = MS2_Parser._seperate_line(line)
                    result["FRAG_MASS"].append(float(mass_frag))
                    result["FRAG_INTENSITY"].append(float(frag_intensity))
        return OUTPUT

    def _msp_parser(file_in):
        """
        Output: list of dictionaries sturcutres as follows:
                [{'Mass': float, 'RT': float, 'CHARGE': str, 'FRAG_MASS': [float], 'FRAG_INTENSITY':[float]}, ...]

                Each list entry represents a feature present within the MS2 dataset
        """
        OUTPUT = []
        with Open_Input(file_in) as file:
            all_lines = file.readlines()
            non_blank_lines = [
                line for line in all_lines if line.strip()
            ]  # Remove empty lines from the input text lines
            has_fragments = True  # This tracks whether a fragment has peaks or not. Initialize as true
            for line in all_lines:
                line = line.strip()  # Get rid of potential blank spaces at end of line
                if line.startswith("Name:"):
                    result = {
                        "MASS": None,
                        "RT": None,
                        "CHARGE": None,
                        "FRAG_MASS": [],
                        "FRAG_INTENSITY": [],
                        "filename": file.filename,
                    }
                elif line.startswith("PrecursorMZ:"):
                    result["MASS"] = float(line.split(" ")[1])
                elif line.startswith("Comment:"):  # RT is stored in the comment line for Waters MSP files
                    line = line.split(" ")[1]
                    result["RT"] = float(line.split("_")[0])
                elif line.startswith("Charge"):
                    result["CHARGE"] = line.split(" ")[1]
                elif line.strip() == "":
                    if has_fragments == True:  # If there are fragment peaks in results, append to OUTPUT
                        OUTPUT.append(result)
                    else:
                        has_fragments = True
                elif line.startswith("Num Peaks:"):  # Check if there are fragments in the spectrum
                    line = line.split(": ")[1]
                    if line == "0":
                        has_fragments = False
                elif line[0].isdigit():  # Only grab fragment data if the line starts with a numeric value
                    mass_frag, frag_intensity = MS2_Parser._seperate_line(line)
                    result["FRAG_MASS"].append(float(mass_frag))
                    result["FRAG_INTENSITY"].append(float(frag_intensity))
                # Add check for last line in file
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
            return line.split("\t")


class Open_Input(object):
    def __init__(self, file_in):
        if isinstance(file_in, str):
            self.file_obj = open(file_in, "r")
            # Get the input filename
            # self.filename = file_in.split("/").pop()
            self.filename = "temp_filename.xxx"
            logger.info(f"filename: {self.filename}")
        else:
            decoded_file = file_in.read().decode("utf-8")
            self.file_obj = io.StringIO(decoded_file)
            # Get the input filename
            self.filename = "temp_filename.xxx"

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
