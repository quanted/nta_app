
from __future__ import absolute_import
import re
import pandas as pd
from operator import itemgetter
from itertools import groupby
from difflib import SequenceMatcher


#convert the user-supplied input file into dataframe
def input_handler(file, index):
    #ext = os.path.splitext(file)[1]
    #print(ext)
    ext = '.csv'  #for now only take csv
    if ext == '.tsv':
        df = pd.read_csv(file, sep='\t', comment='#', na_values=1 | 0)
    if ext == '.csv':
        df = pd.read_csv(file, comment='#', na_values=1 | 0)
                
    #AC
    print("df columns pre-fix names", df.columns)
        
    df = fix_names(df, index)
    
    #AC
    print("df columns pre-fix names", df.columns)
    
    return df


def tracer_handler(file):
    return pd.read_csv(file,comment='#',na_values= 1 | 0)


######## file reader utilities ##########


# format the input dataframe columns
def fix_names(df,index): # parse the Dataframe into a numpy array
        #df.columns = df.columns.str.replace(': Log2','') #log specific code
        df.columns = df.columns.str.replace(' ','_')
        df.columns = df.columns.str.replace('\([^)]*\)','')
        # NTAW-94 comment out the following line. Compound is no longer being used
        # df['Compound'] = df['Compound'].str.replace("\ Esi.*$","")
        if 'Ionization_mode' in df.columns:
            df.rename(columns = {'Ionization_mode':'Ionization_Mode'},inplace=True)
        #df.drop(['CompositeSpectrum','Compound_Name'],axis=1)
        
        #AC 12/12/2023 - I believe the below code is deprecated and is unintentionally renaming samples when there is a large shared string between multiple sample groups
        # if 'Compound_Name' in df.columns:
        #     df.drop(['Compound_Name'],axis=1)
        # Headers = parse_headers(df,index)
        # Abundance = [item for sublist in Headers for item in sublist if len(sublist)>1]
        # Samples= [x for x in Abundance]
        # NewSamples = common_substrings(Samples)
        # df.drop([col for col in df.columns if 'Spectrum' in col], axis=1,inplace=True)
        # for i in range(len(Samples)):

        return df


def parse_headers(df,index): #group headers into a group of samples
        #global df
        headers = [[],[]]
        headers[index] = df.columns.values.tolist()
        countS=0
        countD=0
        new_headers = [[],[]]
        New_Headers = [None,None]
        Headers = [None,None]
        groups = [None,None]
        for s in range(0,len(headers[index])-1):
            #print headers[s],headers[s+1],list(set(str(headers[s])) - set(str(headers[s+1])))
            if 'blank' or 'Blank' or 'MB' in headers[index][s]:
                if differences(str(headers[index][s]),str(headers[index][s+1])) < 2: #3 is more common
                            countS += 1
                if differences(str(headers[index][s]),str(headers[index][s+1])) >= 2:
                            countD += 1
                            countS = countS + 1
            else:
                if differences(str(headers[index][s]),str(headers[index][s+1])) < 2: #2 is more common
                            countS += 1
                if differences(str(headers[index][s]),str(headers[index][s+1])) >= 2:

                            countD += 1
                            countS = countS + 1
                    #print "These are different "
            if "_Flags" in headers[index][s]:
                break
            new_headers[index].append([headers[index][countS],countD])
            new_headers[index].sort(key = itemgetter(1))
            groups[index] = groupby(new_headers[index], itemgetter(1))
            New_Headers[index] = [[item[0] for item in data] for (key, data) in groups[index]]
        Headers[index] = New_Headers[index]
        #print((Headers[1]))
        return Headers[index]


def common_substrings(ls=None):
        match  = SequenceMatcher(None,ls[0],ls[len(ls)-1]).find_longest_match(0,len(ls[0]),0,len(ls[len(ls)-1]))
        common = ls[0][match.a: match.a + match.size]
        #print((" ********* " + common))
        lsnew = list()
        for i in range(len(ls)):
            if len(common) > 3:
                lsnew.append(ls[i].replace(common,''))
            else:
                lsnew.append(ls[i])
        #print ls
        return lsnew


def differences(s1,s2): #find the number of different characters between two strings (headers)
        s1 = re.sub(re.compile(r'\([^)]*\)'),'',s1)
        s2 = re.sub(re.compile(r'\([^)]*\)'),'',s2)
        count = sum(1 for a, b in zip(s1, s2) if a != b) + abs(len(s1) - len(s2))
        return count
