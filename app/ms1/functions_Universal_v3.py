# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:11:13 2017
Modified Nov 18 - Apr 19
Modified Nov 18 - Apr 19

@author: HALGhoul
         Adapted/edited by Jeff Minucci
"""

import pandas as pd
import numpy as np
import re
import os
from operator import itemgetter
from itertools import groupby
from difflib import SequenceMatcher
import dask.dataframe as dd

pd.options.mode.chained_assignment = None  # suppress settingwithcopy warning
"""
#REP_NUM = 3
HBR = 3.0 # High_Blank_Ratio condition
HMR = 1.5 # High_Mid_Ratio condition
SCORE = 90 # formula match is 90
"""
BLANKS = ["MB_", "blank", "blanks", "BLANK", "Blank"]


# def common_substrings(ls=None):
#    match  = SequenceMatcher(None,ls[0],ls[len(ls)-1]).find_longest_match(0,len(ls[0]),0,len(ls[len(ls)-1]))
#    common = ls[0][match.a: match.a + match.size]
#    #print((" ********* " + common))
#    lsnew = list()
#    for i in range(len(ls)):
#        if len(common) > 3:
#            lsnew.append(ls[i].replace(common,''))
#        else:
#            lsnew.append(ls[i])
#            #print ls
#    return lsnew


# def fix_names(df,index): # parse the Dataframe into a numpy array
#        #df.columns = df.columns.str.replace(': Log2','') #log specific code
#        df.columns = df.columns.str.replace(' ','_')
#        df.columns = df.columns.str.replace('\([^)]*\)','')
#        df['Compound'] = df['Compound'].str.replace("\ Esi.*$","")
#        if 'Ionization_mode' in df.columns:
#            df.rename(columns = {'Ionization_mode':'Ionization_Mode'},inplace=True)
#        #df.drop(['CompositeSpectrum','Compound_Name'],axis=1)
#        df.drop(['Compound_Name'],axis=1)
#        Headers = parse_headers(df,index)
#        Abundance = [item for sublist in Headers for item in sublist if len(sublist)>1]
#        Samples= [x for x in Abundance]
#        NewSamples = common_substrings(Samples)
#        df.drop([col for col in df.columns if 'Spectrum' in col], axis=1,inplace=True)
#        for i in range(len(Samples)):
#            df.rename(columns = {Samples[i]:NewSamples[i]},inplace=True)
#        #df = df
#        return df


# def read_data(file,index):  # read a csv file into a DataFrame
#        ext = os.path.splitext(file)[1]
#        #print(ext)
#        if ext == '.tsv':
#            df = pd.read_csv(file,sep='\t',comment='#',na_values= 1 | 0)
#        if ext == '.csv':
#            df = pd.read_csv(file,comment='#',na_values= 1 | 0)
#        df = fix_names(df,index)
#        return df


# def differences(s1,s2): #find the number of different characters between two strings (headers)
#        s1 = re.sub(re.compile(r'\([^)]*\)'),'',s1)
#        s2 = re.sub(re.compile(r'\([^)]*\)'),'',s2)
#        count = sum(1 for a, b in zip(s1, s2) if a != b) + abs(len(s1) - len(s2))
#        return count
