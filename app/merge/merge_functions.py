###########################################################################
# Written for Placenta data
# This code reads in three separate sources of results:
# 1) WebApp MS1 results (df_ms1)
# 2) Reference library serach results (df_pcdl)
# 3) CFM-ID MS2 results (df_cfmid)
# 
# Once the sets are read in, it converts all three dataframes into lists
# For faster matching. It matches the MS1 results against MS2 CFMID and MS2 PCDL
#
# The resulting match information is concatenated (i.e. multiple MS2 files matched to a MS1 feature/chemicals)
# And then merged back onto the initial dataframe of results from the WebApp
# i.e. the final result is the same exact file from MS1 WebApp, with extra columns appended on for
# the match information.
###########################################################################


import pandas as pd
import glob
import numpy as np


