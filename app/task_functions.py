import pandas as pd
import numpy as np
from operator import itemgetter
from itertools import groupby
import os
import re

BLANKS = ['MB_', 'blank', 'blanks', 'BLANK', 'Blank']


def assign_feature_id(df_in, start = 1):
    """
    A function to assign unique feature ids to a nta dataset
    :param df_in: the dataframe to assign ids to
    :param start: assign ids starting at this integer
    :return: returns the new df with unique feature ids added
    """
    df = df_in.copy()
    row_nums = list(range(0, len(df.index)))
    to_assign = [x + start for x in row_nums]
    df.insert(0, 'Feature_ID', to_assign.copy())
    return df


def differences(s1, s2):
    """
    find the number of different characters between two strings (headers)
    """
    s1 = re.sub(re.compile(r'\([^)]*\)'), '', s1)
    s2 = re.sub(re.compile(r'\([^)]*\)'), '', s2)
    count = sum(1 for a, b in zip(s1, s2) if a != b) + abs(len(s1) - len(s2))
    return count


def formulas(df):
    df.drop_duplicates(subset='Compound',keep='first',inplace=True)
    formulas = df.loc[df['For_Dashboard_Search'] == '1','Compound'].values
    formulas_list = [str(i) for i in formulas]
    return formulas_list


def masses(df):
    #df.drop_duplicates(subset='Mass',keep='first',inplace=True)  # TODO should this be on?
    masses = df.loc[df['For_Dashboard_Search'] == '1','Mass'].values
    masses_list = [str(i) for i in masses]
    return masses_list


#untested
def parse_headers(df_in):
    '''
    A function to group the dataframe's column headers into sets of similar names which represent replicates
    :param df_in: the dataframe of features
    :return: a list of groups of column labels
    '''
    df = df_in.copy()
    headers = df.columns.values.tolist()
    countS = 0
    countD = 0
    new_headers = []
    for s in range(0,len(headers)-1):
        if 'blank' or 'Blank' or 'MB' in headers[s]:
            if differences(str(headers[s]),str(headers[s+1])) < 2: #3 is more common
                countS += 1
            if differences(str(headers[s]),str(headers[s+1])) >= 2:
                countD += 1
                countS = countS + 1
        else:
            if differences(str(headers[s]),str(headers[s+1])) < 2: #2 is more common
                countS += 1
            if differences(str(headers[s]),str(headers[s+1])) >= 2:

                countD += 1
                countS = countS + 1
            #print "These are different "
        if "_Flags" in headers[s]:
            break
        new_headers.append([headers[countS],countD])
        new_headers.sort(key = itemgetter(1))
    groups = groupby(new_headers, itemgetter(1))
    new_headers_list = [[item[0] for item in data] for (key, data) in groups]
    return new_headers_list



def adduct_identifier(df_in, Mass_Difference, Retention_Difference, ppm, ionization, id_start = 1):  # TODO optimize memory usage
    df = df_in.copy()
    mass = df['Mass'].to_numpy()
    rts = df['Retention_Time'].to_numpy()
    masses_matrix = np.reshape(mass, (len(mass), 1))
    rts_matrix = np.reshape(rts, (len(rts),1))
    diff_matrix_mass = masses_matrix - masses_matrix.transpose()
    diff_matrix_rt = rts_matrix - rts_matrix.transpose()
    pos_adduct_deltas = {'Na': 22.989218, 'K': 38.963158, 'NH4': 18.033823}
    neg_adduct_deltas = {'Cl': 34.969402, 'Br': 78.918885, 'HCO2': 44.998201, 'CH3CO2': 59.013851, 'CF3CO2': 112.985586}
    neutral_loss_deltas= {'H2O': -18.010565, 'CO2': -43.989829}
    proton_mass = 1.007276
    if ionization == "positive":
        # we observe Mass+(H+) and Mass+(Adduct)
        possible_adduct_deltas = {k: v - proton_mass for (k,v) in pos_adduct_deltas.items()}
    else:
        # we observe Mass-(H+) and Mass+(Adduct)
        possible_adduct_deltas = {k: v + proton_mass for (k,v) in neg_adduct_deltas.items()}
    possible_adduct_deltas.update(neutral_loss_deltas)  # add our neutral losses
    df['Has_Adduct_or_Loss'] = 0
    df['Is_Adduct_or_Loss'] = 0
    df['Adduct_or_Loss_Info'] = ""
    unique_adduct_number = np.zeros(len(df.index))
    for a_name, delta in sorted(possible_adduct_deltas.items()):
        is_adduct_diff = abs(diff_matrix_mass - delta)
        has_adduct_diff = abs(diff_matrix_mass + delta)
        if ppm:
            is_adduct_diff = (is_adduct_diff/masses_matrix)*10**6
            has_adduct_diff = (has_adduct_diff/masses_matrix)*10**6
        is_adduct_matrix = np.where((is_adduct_diff < Mass_Difference) & (abs(diff_matrix_rt) < Retention_Difference), 1, 0)
        has_adduct_matrix = np.where((has_adduct_diff < Mass_Difference) & (abs(diff_matrix_rt) < Retention_Difference), 1, 0)
        np.fill_diagonal(is_adduct_matrix, 0)  # remove self matches
        np.fill_diagonal(has_adduct_matrix, 0)  # remove self matches
        row_num = len(mass)
        is_id_matrix = np.tile(np.arange(row_num),row_num).reshape((row_num,row_num)) + id_start
        has_id_matrix = is_id_matrix.transpose()
        is_adduct_number = is_adduct_matrix * is_id_matrix
        is_adduct_number_flat = np.max(is_adduct_number, axis=1) # if is adduct of multiple, keep highest # row
        is_adduct_number_flat_index = np.where(is_adduct_number_flat > 0, is_adduct_number_flat -1, 0)
        is_adduct_of_adduct = np.where((is_adduct_number_flat > 0) &
                                       (df['Is_Adduct_or_Loss'][is_adduct_number_flat_index] > 0), 1, 0)
        is_adduct_number_flat[is_adduct_of_adduct == 1] = 0
        has_adduct_number = has_adduct_matrix * is_id_matrix
        has_adduct_number_flat = np.max(has_adduct_number, axis=1)  # these will all be the same down columns
        unique_adduct_number = np.where(has_adduct_number_flat != 0, has_adduct_number_flat, is_adduct_number_flat).astype(int)
        #unique_adduct_number = np.where(unique_adduct_number == 0, unique_adduct_number_new, unique_adduct_number)
        df['Has_Adduct_or_Loss'] = np.where((has_adduct_number_flat > 0) & (df['Is_Adduct_or_Loss'] == 0),
                                            df['Has_Adduct_or_Loss']+1, df['Has_Adduct_or_Loss'])
        df['Is_Adduct_or_Loss'] = np.where((is_adduct_number_flat > 0) & (df['Has_Adduct_or_Loss'] == 0), 1, df['Is_Adduct_or_Loss'])
        # new_cols = ['unique_{}_number'.format(a_name), 'has_{}_adduct'.format(a_name), 'is_{}_adduct'.format(a_name)]
        #unique_adduct_number_str =
        df['Adduct_or_Loss_Info'] = np.where((has_adduct_number_flat > 0) & (df['Is_Adduct_or_Loss'] == 0),
                                             df['Adduct_or_Loss_Info'] + unique_adduct_number.astype(str) + "({});".format(a_name), df['Adduct_or_Loss_Info'])
        df['Adduct_or_Loss_Info'] = np.where((is_adduct_number_flat > 0) & (df['Has_Adduct_or_Loss'] == 0),
                                             df['Adduct_or_Loss_Info'] + unique_adduct_number.astype(str) + "({});".format(a_name), df['Adduct_or_Loss_Info'])
    return df
