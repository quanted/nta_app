import pandas as pd
import numpy as np
from operator import itemgetter
from difflib import SequenceMatcher
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
    """
    Label features which could have adduct or loss products in the feature list, or may be adduct of loss products of
    other features. This information is added to the dataframe in three columns, Has_Adduct_or_Loss - true/false,
    Is_Adduct_or_Loss - true/false, and Adduct_or_Less_Info which points to the feature number of the associated
    adduct/loss or parent feature and gives the type of adduct/loss.
    :param df_in: A dataframe of MS features
    :param Mass_Difference: Mass differences below this number are possible duplicates. Units of ppm or Da based on the
    ppm parameter.
    :param Retention_Difference: Retention time differences below this number are possible duplicates. Units of mins.
    :param ppm: True if mass differences are given in ppm, otherwise False and units are Da.
    :param ionization: 'positive' if data are from positive ionization mode, otherwise 'negative'.
    :param id_start: The first feature id in the dataset (defaults to 1)
    :return: A Dataframe where adduct info is given in three new columns.
    """
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
                                       (df['Is_Adduct_or_Loss'][pd.Series(is_adduct_number_flat_index-id_start).clip(lower=0)] > 0), 1, 0)
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


#untested
def duplicates(df, mass_cutoff=0.005, rt_cutoff=0.05):
    """
    Drop features that are deemed to be duplicates. Duplicates are defined as two features whose differences in
    both mass and retention time are less than the defined cutoffs.
    :param df: A dataframe of MS feature data
    :param mass_cutoff: Mass differences below this number are possible duplicates. Units of Da.
    :param rt_cutoff: Retention time differences below this number are possible duplicates. Units of mins.
    :return: A dataframe with duplicates removed
    """
    df_new = df.copy()
    samples_df = df.filter(like='Sample', axis=1)
    df_new['all_sample_mean'] = samples_df.mean(axis=1)  # mean intensity across all samples
    df_new.sort_values(by=['all_sample_mean'], inplace=True, ascending=False)
    df_new.reset_index(drop=True, inplace=True)
    mass = df_new['Mass'].to_numpy()
    rts = df_new['Retention_Time'].to_numpy()
    masses_matrix = np.reshape(mass, (len(mass), 1))
    rts_matrix = np.reshape(rts, (len(rts), 1))
    diff_matrix_mass = masses_matrix - masses_matrix.transpose()
    diff_matrix_rt = rts_matrix - rts_matrix.transpose()
    duplicates_matrix = np.where((abs(diff_matrix_mass) <= mass_cutoff) & (abs(diff_matrix_rt) <= rt_cutoff),1,0)
    np.fill_diagonal(duplicates_matrix, 0)
    row_sums = np.sum(duplicates_matrix, axis=1)  # gives number of duplicates for each df row
    duplicates_matrix_lower = np.tril(duplicates_matrix)  # lower triangle of matrix
    lower_row_sums = np.sum(duplicates_matrix_lower, axis=1)
    to_keep = df_new[(row_sums == 0) | (lower_row_sums == 0)].copy()
    to_keep.sort_values(by=['Mass'], inplace=True)
    to_keep.reset_index(drop=True, inplace=True)
    to_keep = to_keep.drop(['all_sample_mean'], axis=1).copy()
    return to_keep


#untested
def statistics(df_in):
    """
    # Calculate Mean,Median,STD,CV for every feature in a sample of multiple replicates
    :param df_in: the dataframe to calculate stats for
    :return: a new dataframe including the stats
    """
    df = df_in.copy()
    all_headers = parse_headers(df)
    abundance = [item for sublist in all_headers for item in sublist if len(sublist) > 1]
    df = score(df)
    filter_headers= ['Compound','Ionization_Mode','Score','Mass','Retention_Time','Frequency'] + abundance
    df = df[filter_headers].copy()
    for the_list in all_headers:
        REP_NUM = len(the_list)
        if REP_NUM > 1:
            for i in range(0, REP_NUM):
                # match finds the indices of the largest common substring between two strings
                match = SequenceMatcher(None, the_list[i], the_list[i+1]).find_longest_match(0, len(the_list[i]),0, len(the_list[i+1]))
                df['Mean_'+ str(the_list[i])[match.a:match.a +  match.size]] = df[the_list[i:i + REP_NUM]].mean(axis=1).round(0)
                df['Median_'+ str(the_list[i])[match.a:match.a +  match.size]] = df[the_list[i:i + REP_NUM]].median(axis=1,skipna=True).round(0)
                df['STD_'+ str(the_list[i])[match.a:match.a +  match.size]] = df[the_list[i:i + REP_NUM]].std(axis=1,skipna=True).round(0)
                df['CV_'+ str(the_list[i])[match.a:match.a +  match.size]] = (df['STD_'+ str(the_list[i])[match.a:match.a +  match.size]]/df['Mean_' + str(the_list[i])[match.a:match.a + match.size]]).round(2)
                df['N_Abun_'+ str(the_list[i])[match.a:match.a +  match.size]] = df[the_list[i:i + REP_NUM]].count(axis=1).round(0)
                break
    df.sort_values(['Mass', 'Retention_Time'], ascending=[True, True], inplace=True)
    return df


def score(df):  # Get score from annotations.
    regex = "^.*=(.*) \].*$"  # a regex to find the score looking for a pattern of "=something_to_find ]"
    if "Annotations" in df:
        if df.Annotations.isnull().all():  # make sure there isn't a totally blank Annotations column
            df['Score'] = None
            return df
        if df.Annotations.str.contains('overall=').any():
            df['Score'] = df.Annotations.str.extract(regex, expand=True).astype('float64')
    elif "Score" in df:
        pass
    else:
        df['Score'] = None
    return df
