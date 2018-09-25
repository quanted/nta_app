import pandas as pd
import numpy as np
import json
import gridfs
import sys
from datetime import datetime
from .functions_Universal_v3 import parse_headers, duplicates, statistics,\
    adduct_identifier, check_feature_tracers, clean_features
from .utilities import connect_to_mongoDB


class NtaRun:

    def __init__(self, parameters, input_dfs, tracer_df = None, jobid = "00000000", verbose = True):
        self.mass_accuracy = float(parameters['mass_accuracy'])
        self.mass_accuracy_units = parameters['mass_accuracy_units']
        self.rt_accuracy = float(parameters['rt_accuracy'])
        self.tracer_df = tracer_df
        self.tracer_dfs_out = None
        self.mass_accuracy_tr = float(parameters['mass_accuracy_tr'])
        self.mass_accuracy_units_tr = parameters['mass_accuracy_units_tr']
        self.rt_accuracy_tr = float(parameters['rt_accuracy_tr'])
        self.entact = parameters['entact'] == "yes"
        self.sample_to_blank = float(parameters['sample_to_blank'])
        self.min_replicate_hits = float(parameters['min_replicate_hits'])
        self.max_replicate_cv = float(parameters['max_replicate_cv'])
        self.parent_ion_mass_accuracy = float(parameters['parent_ion_mass_accuracy'])
        self.search_mode = parameters['search_mode']
        self.top_result_only = parameters['top_result_only'] == 'yes'
        self.dfs = input_dfs
        self.jobid = jobid
        self.verbose = verbose
        self.mongo = connect_to_mongoDB()


    def execute(self):
        print(self.entact)
        # 1: drop duplicates
        self.drop_duplicates()
        if self.verbose:
            print("Dropped duplicates.")
            #print(self.dfs[0])

        # 2: statistics
        self.calc_statistics()
        if self.verbose:
            print("Calculated statistics.")
            #print(self.dfs[0])
            #print(str(list(self.dfs[0])))

        # 3: check tracers (optional)
        self.check_tracers()
        if self.verbose:
            print("Checked tracers.")
            #print(self.tracer_dfs_out)

        # 4: clean features
        self.clean_features()
        if self.verbose:
            print("Cleaned features.")
            print(self.dfs[0])


    def drop_duplicates(self):
        self.dfs = [duplicates(df, index) for index, df in enumerate(self.dfs)]
        #self.mongo_save(self.dfs[0], 'input_no_duplicates_pos')
        #self.mongo_save( self.dfs[1], 'input_no_duplicates_neg')
        return

    def calc_statistics(self):
        ppm = self.mass_accuracy_units == 'ppm'
        self.dfs = [statistics(df, index) for index, df in enumerate(self.dfs)]
        print("Calculating statistics with units: " + self.mass_accuracy_units)
        self.dfs = [adduct_identifier(df, index, self.mass_accuracy, self.rt_accuracy, ppm) for index, df in enumerate(self.dfs)]
        #self.save_df_to_mongo('stats_pos', self.dfs[0])
        #self.save_df_to_mongo('stats_neg', self.dfs[1])
        self.mongo_save(self.dfs[0], 'stats_pos')
        self.mongo_save(self.dfs[1], 'stats_neg')
        return


    def check_tracers(self):
        if self.tracer_df is None:
            print("No tracer file, skipping this step.")
            return
        if self.verbose:
            print("Tracer file found, checking tracers.")
        ppm = self.mass_accuracy_units_tr == 'ppm'
        self.tracer_dfs_out = [check_feature_tracers(df, self.tracer_df, self.mass_accuracy_tr, self.rt_accuracy_tr, ppm) for index, df in enumerate(self.dfs)]
        self.mongo_save(self.tracer_dfs_out[0], 'tracers_pos')
        self.mongo_save(self.tracer_dfs_out[1], 'tracers_neg')
        return

    def clean_features(self):
        controls = [self.sample_to_blank, self.min_replicate_hits, self.max_replicate_cv]
        self.dfs = [clean_features(df, index, self.entact, controls) for index, df in enumerate(self.dfs)]
        self.mongo_save(self.dfs[0], 'cleaned_pos')
        self.mongo_save(self.dfs[1], 'cleaned_neg')
        print(self.dfs[0])
        return


    def mongo_save(self, file, step=""):
        to_save = json.loads(file.to_json(orient='index'))
        posts = self.mongo.posts
        time_stamp = datetime.utcnow()
        id = self.jobid + "_" + step
        data = {'_id': id, 'date': time_stamp, 'data': to_save}
        posts.insert_one(data)


    # def save_df_to_mongo(self, label, df): #TODO: save each csv as a new item, save raw csv as binary not json
    #     posts = self.mongo.posts
    #     time_stamp = datetime.utcnow()
    #     data = df.to_json(orient='index')
    #     posts.update_one({'_id': self.jobid}, {'$set': {label: data}})
