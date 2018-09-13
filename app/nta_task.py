import pandas as pd
import numpy as np
from .functions_Universal_v3 import parse_headers, duplicates, statistics, adduct_identifier
from ..tools.file_manager import connect_to_mongoDB


class NtaRun:

    def __init__(self, parameters, input_dfs, jobid = "00000000", verbose = True):
        self.mass_accuracy = float(parameters['mass_accuracy'])
        self.mass_accuracy_units = parameters['mass_accuracy_units']
        self.rt_accuracy = float(parameters['rt_accuracy'])
        self.dfs = input_dfs
        self.jobid = jobid
        self.verbose = verbose


    def execute(self):
        # 1: drop duplicates
        self.drop_duplicates()
        if self.verbose:
            print("Dropped duplicates.")
            #print(self.dfs[0])

        # 2: statistics
        self.calc_statistics()
        if self.verbose:
            print("Calculated statistics.")
            print(self.dfs[0])


    def drop_duplicates(self):
        self.dfs = [duplicates(df, index) for index, df in enumerate(self.dfs)]
        return

    def calc_statistics(self):
        ppm = self.mass_accuracy_units == 'ppm'
        self.dfs = [statistics(df, index) for index, df in enumerate(self.dfs)]
        print("Calculating statistics with units: " + self.mass_accuracy_units)
        self.dfs = [adduct_identifier(df, index, self.mass_accuracy, self.rt_accuracy, ppm) for index, df in enumerate(self.dfs)]
        return







