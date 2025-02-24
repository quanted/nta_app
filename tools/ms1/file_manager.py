from __future__ import absolute_import
import re
import pandas as pd
from operator import itemgetter
from itertools import groupby
from difflib import SequenceMatcher


# convert the user-supplied input file into dataframe
def input_handler(file, index, na_value):
    # ext = os.path.splitext(file)[1]
    # print(ext)
    # Hard-coded to only accept .csv files
    ext = ".csv"
    if ext == ".tsv":
        df = pd.read_csv(file, sep="\t", comment="#", na_values=na_value)
    if ext == ".csv":
        # Read .csv file, add user-selected na_value to list of default na values for pandas na filter
        df = pd.read_csv(file, comment="#", na_values=na_value, keep_default_na=True, na_filter=True)
    # Call fix names
    df = fix_names(df, index)
    # Return formatted df
    return df


def tracer_handler(file):
    return pd.read_csv(file, comment="#", na_values=1 | 0)


######## file reader utilities ##########


# format the input dataframe columns
def fix_names(df, index):  # parse the Dataframe into a numpy array
    # df.columns = df.columns.str.replace(': Log2','') #log specific code
    df.drop(df.columns[df.columns.str.startswith("Unnamed: ")], axis=1, inplace=True)
    df.columns = df.columns.str.replace(" ", "_")
    df.columns = df.columns.str.replace("#", "_")
    df.columns = df.columns.str.replace("\([^)]*\)", "")
    # NTAW-94 comment out the following line. Compound is no longer being used
    # df['Compound'] = df['Compound'].str.replace("\ Esi.*$","")
    if "Ionization_mode" in df.columns:
        df.rename(columns={"Ionization_mode": "Ionization_Mode"}, inplace=True)
    # df.drop(['CompositeSpectrum','Compound_Name'],axis=1)

    # AC 12/12/2023 - I believe the below code is deprecated and is unintentionally renaming samples when there is a large shared string between multiple sample groups
    # if 'Compound_Name' in df.columns:
    #     df.drop(['Compound_Name'],axis=1)
    # Headers = parse_headers(df,index)
    # Abundance = [item for sublist in Headers for item in sublist if len(sublist)>1]
    # Samples= [x for x in Abundance]
    # NewSamples = common_substrings(Samples)
    # df.drop([col for col in df.columns if 'Spectrum' in col], axis=1,inplace=True)
    # for i in range(len(Samples)):

    return df


def common_substrings(ls=None):
    match = SequenceMatcher(None, ls[0], ls[len(ls) - 1]).find_longest_match(0, len(ls[0]), 0, len(ls[len(ls) - 1]))
    common = ls[0][match.a : match.a + match.size]
    # print((" ********* " + common))
    lsnew = list()
    for i in range(len(ls)):
        if len(common) > 3:
            lsnew.append(ls[i].replace(common, ""))
        else:
            lsnew.append(ls[i])
    # print ls
    return lsnew


# def differences(s1, s2):  # find the number of different characters between two strings (headers)
#     s1 = re.sub(re.compile(r"\([^)]*\)"), "", s1)
#     s2 = re.sub(re.compile(r"\([^)]*\)"), "", s2)
#     count = sum(1 for a, b in zip(s1, s2) if a != b) + abs(len(s1) - len(s2))
#     return count
