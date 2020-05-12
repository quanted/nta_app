# -*- coding: utf-8 -*-
"""
Created on Tue May 01 12:34:34 2018

@author: HALGhoul
"""

import mgf_parser_v24_AC as mg
import time
import os
import glob

path = "mgf/"
# This loops through all mgf files and generates csv files
for infile in glob.glob( os.path.join(path, '*.mgf') ): # Only reads mgf files in a directory

    print "MGF to CSV: current file is: " + infile
    mg.parseMGF(infile) # parse the file


fpcdl = os.getcwd()+'/500 master spiked list.csv' # This is the list of masses to input that it will look for in the mgf/csv file
# CFMID search each csv file generated from above
path = "mgf/"
for infile in glob.glob( os.path.join(path, '*.csv') ): # Reads all CSV files in directory
    print "CSV search/score: current file is: " + infile
    t0=time.clock()
    mg.compare_mgf_df(infile,infile,10,0.02,POSMODE=True,filtering=False)  # filtering always false (for now)
    t1=time.clock()
    print("time to Process is " + str(t1-t0))
    #print("Total time processing is " + str(round((t1-t_start)/60)) + " minutes")




