import pandas as pd
import numpy as np
from operator import itemgetter
from difflib import SequenceMatcher
from itertools import groupby
import os
import re
import logging

#  BLANKS = ['MB_', 'blank', 'blanks', 'BLANK', 'Blank']  NOT USED

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("nta_app.ms1")
logger.setLevel(logging.INFO)

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
    #df.drop_duplicates(subset='Mass', keep='first',inplace=True)  # TODO should this be on?
    masses = df.loc[df['For_Dashboard_Search'] == '1','Mass'].values
    logger.info('# of masses for dashboard search: {} out of {}'.format(len(masses),len(df)))
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


# Called within the 'duplicates' function - takes a filtered 'to_test' df, does matrix math, returns 'passed'
def dup_matrix(df_in, mass_cutoff, rt_cutoff):
    logger.info("Checkpoint i")
    # Create matrices from df_in
    mass = df_in['Mass'].to_numpy()
    rts = df_in['Retention_Time'].to_numpy()
    logger.info("Checkpoint ii")
    # Reshape matrices
    masses_matrix = np.reshape(mass, (len(mass), 1))
    rts_matrix = np.reshape(rts, (len(rts), 1))
    logger.info("Checkpoint iii")
    # Perform matrix transposition
    diff_matrix_mass = masses_matrix - masses_matrix.transpose()
    diff_matrix_rt = rts_matrix - rts_matrix.transpose()
    logger.info("Checkpoint iv")
    # Find indices where differences are less than 'mass_cutoff' and 'rt_cutoff'
    duplicates_matrix = np.where((abs(diff_matrix_mass) <= mass_cutoff) & (abs(diff_matrix_rt) <= rt_cutoff),1,0)
    np.fill_diagonal(duplicates_matrix, 0)
    logger.info("Checkpoint v")
    # Find # of duplicates for each row
    row_sums = np.sum(duplicates_matrix, axis=1)
    logger.info("Checkpoint vi")
    # Calculate lower triangle of matrix
    duplicates_matrix_lower = np.tril(duplicates_matrix)
    logger.info("Checkpoint vii")
    lower_row_sums = np.sum(duplicates_matrix_lower, axis=1)
    logger.info("Checkpoint viii")
    # Store features with no duplicates in 'passed'
    passed = df_in[(row_sums == 0) | (lower_row_sums == 0)].copy()
    logger.info("Checkpoint ix")
    
    return passed
    
    
def duplicates(df_in, mass_cutoff=0.005, rt_cutoff=0.05):
    #tracemalloc.start()
    #start = time.perf_counter()
    logger.info("Checkpoint a")
    # Copy the dataframe
    df = df_in.copy()
    # Parse headers to find sample columns
    all_headers = parse_headers(df) 
    sam_headers = [item for sublist in all_headers for item in sublist if len(sublist) > 1]
    logger.info("Checkpoint b")
    # Create 'Rounded Mass' and 'Rounded RT' columns
    df['Rounded Mass'] = df['Mass'].round(2)
    df['Rounded RT'] = df['Retention_Time'].round(1)
    logger.info("Checkpoint c")
    # Calculate 'all_sample_mean', sort df by 'all_sample_mean', reset index
    df['all_sample_mean'] = df[sam_headers].mean(axis=1)  # mean intensity across all samples
    df.sort_values(by=['all_sample_mean'], inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)
    logger.info("Checkpoint d")
    # Filter df into 'to_keep' (all uniques), and 'to_test' (df indices not in 'to_keep')
    to_keep = df.drop_duplicates(subset=['Rounded Mass', 'Rounded RT'], keep =False)
    to_test = df[~df.index.isin(to_keep.index)]
    logger.info("Checkpoint 5")
    # Concatenate 'to_keep' and output of 'dup_matrix' function (see above)
    output = pd.concat([to_keep, dup_matrix(to_test, mass_cutoff, rt_cutoff)], axis=0)
    logger.info("Checkpoint e")
    # Sort output by 'Mass', reset the index, drop 'all_sample_mean'
    output.sort_values(by=['Mass'], inplace=True)
    output.reset_index(drop=True, inplace=True)
    #output.drop(['all_sample_mean'], axis=1, inplace=True)
    logger.info("Checkpoint f")
    #finish = time.perf_counter()
    #print(f'Finished in {finish-start} seconds.')

    #print(tracemalloc.get_traced_memory())
    #tracemalloc.stop()
    
    return output


'''
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
'''



'''
#untested
def statistics(df_in):
    """
    # Calculate Mean,Median,STD,CV for every feature in a sample of multiple replicates
    :param df_in: the dataframe to calculate stats for
    :return: a new dataframe including the stats
    """
    # make a copy of the input DataFrame, df_in, to avoid modifying the original DataFrame.
    df = df_in.copy()
    # extract the headers (column names) from the input DataFrame by calling a function called parse_headers. The extracted headers are stored in the all_headers variable.
    all_headers = parse_headers(df)
    # create a list called abundance by flattening the all_headers list and filtering out items with a length greater than 1. 
    abundance = [item for sublist in all_headers for item in sublist if len(sublist) > 1]
    # call the score function to calculate the score for each feature in the input DataFrame. The score is stored in a new column called Score.
    df = score(df)
    # create a list called filter_headers, which includes a predefined set of column names to keep in the DataFrame. 
    filter_headers= ['Compound','Ionization_Mode','Score','Mass','Retention_Time'] + abundance
    # create a new DataFrame called df, which includes only the columns in the filter_headers list.
    df = df[filter_headers].copy()
    
    # 8/16/2023 AC: Adjust code generating new statistics columns to avoid fragmented Dataframe / frame.insert messages
    # For each list of replicates (the_list), it calculates statistics for each replicate by comparing them. Specifically, 
    # it finds the longest common substring between the replicate names, and this common substring is used to create new 
    # columns in the DataFrame to store the statistics (Mean, Median, STD, CV, N_Abun).
    for the_list in all_headers:
        REP_NUM = len(the_list)
        if REP_NUM > 1:
            i = 0
            # match finds the indices of the largest common substring between two strings
            match = SequenceMatcher(None, the_list[i], the_list[i+1]).find_longest_match(0, len(the_list[i]),0, len(the_list[i+1]))
            df['Mean_'+ str(the_list[i])[match.a:match.a +  match.size]] = df[the_list[i:i + REP_NUM]].mean(axis=1).round(4)
            df['Median_'+ str(the_list[i])[match.a:match.a +  match.size]] = df[the_list[i:i + REP_NUM]].median(axis=1,skipna=True).round(4)
            df['STD_'+ str(the_list[i])[match.a:match.a +  match.size]] = df[the_list[i:i + REP_NUM]].std(axis=1,skipna=True).round(4)
            df['CV_'+ str(the_list[i])[match.a:match.a +  match.size]] = (df['STD_'+ str(the_list[i])[match.a:match.a +  match.size]]/df['Mean_' + str(the_list[i])[match.a:match.a + match.size]]).round(4)
            df['N_Abun_'+ str(the_list[i])[match.a:match.a +  match.size]] = df[the_list[i:i + REP_NUM]].count(axis=1).round(0)
            # add column named 'Total_Abun_' for the total number of replicates for the sample group
            df['Total_Abun_'+ str(the_list[i])[match.a:match.a +  match.size]] = REP_NUM
            Replicate_Percent = (df['N_Abun_'+ str(the_list[i])[match.a:match.a +  match.size]]/df['Total_Abun_'+ str(the_list[i])[match.a:match.a +  match.size]]).round(4)
            df['Replicate_Percent_'+ str(the_list[i])[match.a:match.a +  match.size]] = Replicate_Percent * 100.0

           
    # 8/21/2023 AC: Adjust code to dictionary comprehension to avoid Dataframe fragmentations
    # for the_list in all_headers:
    #     REP_NUM = len(the_list)
    #     if REP_NUM > 1:
    #         stats_data = {}
    #         for i in range(0, REP_NUM):
    #             # match finds the indices of the largest common substring between two strings
    #             match = SequenceMatcher(None, the_list[i], the_list[i+1]).find_longest_match(0, len(the_list[i]),0, len(the_list[i+1]))
    #             stats_data['Mean_'+ str(the_list[i])[match.a:match.a +  match.size]] = df[the_list[i:i + REP_NUM]].mean(axis=1).round(0)
    #             stats_data['Median_'+ str(the_list[i])[match.a:match.a +  match.size]] = df[the_list[i:i + REP_NUM]].median(axis=1,skipna=True).round(0)
    #             stats_data['STD_'+ str(the_list[i])[match.a:match.a +  match.size]] = df[the_list[i:i + REP_NUM]].std(axis=1,skipna=True).round(0)
    #             stats_data['CV_'+ str(the_list[i])[match.a:match.a +  match.size]] = (stats_data['STD_'+ str(the_list[i])[match.a:match.a +  match.size]]/stats_data['Mean_'+ str(the_list[i])[match.a:match.a +  match.size]]).round(4)
    #             stats_data['N_Abun_'+ str(the_list[i])[match.a:match.a +  match.size]] = df[the_list[i:i + REP_NUM]].count(axis=1).round(0)
    #             break
    #         new_df = pd.concat(stats_data.values(), axis=1, ignore_index=True)
    #         new_df.columns = stats_data.keys()  # since Python 3.7, order of insertion is preserved
    # df = df.join(new_df)
    
    # sort the DataFrame based on the 'Mass' and 'Retention_Time' columns in ascending order.
    df.sort_values(['Mass', 'Retention_Time'], ascending=[True, True], inplace=True)
    # round the 'Mass' column to zero decimal places and stores the result in a new column called 'Rounded_Mass'.
    df['Rounded_Mass'] = df['Mass'].round(0)

    # Create a new dataframe column titled "Max_CV_across_sample" and populate it with the maximum CV value for each feature across all columns containing the string "CV_" in the header
    # df=df.assign(Max_CV_across_sample=df.filter(regex='CV_').max(axis=1))
    df['Max_CV_across_sample'] = df.filter(regex='CV_').max(axis=1)

    return df
    '''
    
def statistics(df_in):
    #tracemalloc.start()
    #start = time.perf_counter()
    df = df_in.copy()

    all_headers = parse_headers(df_in) 
    sam_headers = [i for i in all_headers if len(i) > 1]
    
    mean_cols = ['Mean_' + i[0][:-1] for i in sam_headers]
    med_cols = ['Median_' + i[0][:-1] for i in sam_headers]
    std_cols = ['STD_' + i[0][:-1] for i in sam_headers]
    cv_cols = ['CV_' + i[0][:-1] for i in sam_headers]
    nabun_cols = ['N_Abun_' + i[0][:-1] for i in sam_headers]
    #tabun_cols = ['Total_Abun_' + i[0][:-1] for i in sam_headers]
    rper_cols = ['Replicate_Percent_' + i[0][:-1] for i in sam_headers]

    #attrs = pd.concat([df[x] for x in attr], axis=1)
    means = pd.concat([df[x].mean(axis=1).round(4).rename(col) for x, col in zip(sam_headers, mean_cols)], axis=1)
    medians = pd.concat([df[x].median(axis=1, skipna=True).round(4).rename(col) for x, col in zip(sam_headers, med_cols)], axis=1)
    stds = pd.concat([df[x].std(axis=1, skipna=True).round(4).rename(col) for x, col in zip(sam_headers, std_cols)], axis=1)
    cvs = pd.concat([(stds[scol]/means[mcol]).round(4).rename(col) for mcol, scol, col in zip(mean_cols, std_cols, cv_cols)], axis=1)
    nabuns = pd.concat([df[x].count(axis=1).round(0).rename(col) for x, col in zip(sam_headers, nabun_cols)], axis=1)
    #tabuns = pd.concat([pd.Series(len(x), index = df.index(), name=col) for x, col in zip(sam_headers, tabun_cols)], axis=1)
    rpers = pd.concat([((nabuns[ncol]/len(x)).round(4)*100).rename(col) for x, ncol, col in zip(sam_headers, nabun_cols, rper_cols)], axis=1)

    output = pd.concat([df, means, medians, stds, cvs, nabuns, rpers], axis=1)
    
    #finish = time.perf_counter()
    #print(f'Finished in {finish-start} seconds.')
    
    #print(tracemalloc.get_traced_memory())
    #tracemalloc.stop()
    
    return output
    
    
def chunk_stats(df_in):
    logger.info("Starting chunk stats.")
    # Create copy
    df=df_in.copy()
    # Calculate 'score'
    df = score(df)
    # Set chunk size (i.e., # rows)
    n = 5000
    # Create list of Data.Frame chunks
    list_df = [df[i:i+n] for i in range(0,df.shape[0],n)]
    # Instantiate empty list
    li=[]
    
    logger.info("Starting stats loop.")
    # iterate through list_df, calculating 'statistics' on chunks and appending to li
    for df in list_df:
        li.append(statistics(df))
        logger.info("In stats loop.")
    
    logger.info("Finished stats loop.")
    # concatenate li, sort, and calculate 'Rounded_Mass' + 'Max_CV_across_sample'
    output = pd.concat(li, axis=0)
    output.sort_values(['Mass', 'Retention_Time'], ascending=[True, True], inplace=True)
    output['Rounded_Mass'] = output['Mass'].round(0)
    output['Max_CV_across_sample'] = output.filter(regex='CV_').max(axis=1)
    
    logger.info("Finished chunk stats.")
    return output
    

def cal_detection_count(df_in):
    blanks = ['MB','mb','mB','Mb','blank','Blank','BLANK']

    # make a working copy of the dataframe
    df = df_in.copy()

    # a list of lists of headers that contain abundance data
    all_header_groups = parse_headers(df)
    abundance = [item for sublist in all_header_groups for item in sublist if len(sublist) > 1]
    filter_headers= ['Compound'] + abundance

    # remove all items filter_headers containing string 'BlankSub_Median_' from list filter_headers
    filter_headers = [item for item in filter_headers if 'BlankSub_Median_' not in item]

    filter_headers_nonblanks = [item for item in filter_headers if not any(x in item for x in blanks)]

# Std_samples = [md for md in Std if not any(x in md for x in blanks)]

    df = df[filter_headers].copy()
    df_nonblanks = df[filter_headers_nonblanks].copy()

    # calculate detection_Count
    df['Detection_Count(all_samples)'] = df.count(axis=1)

    # subtract 1 from detection_Count to account for the compound name
    df['Detection_Count(all_samples)'] = df['Detection_Count(all_samples)'].apply(lambda x: x - 1)

    # total number of samples (subtract 1 for the compound name)
    total_samples = len(filter_headers) - 1

    # calculate percentage of samples that have a value and store in new column 'detection_Count(all_samples)(%)'
    df['Detection_Count(all_samples)(%)'] = (df['Detection_Count(all_samples)'] / total_samples) * 100
    # round to whole number
    df['Detection_Count(all_samples)(%)'] = df['Detection_Count(all_samples)(%)'].round(0)




    # calculate non-blank_samples
    df_nonblanks['Detection_Count(non-blank_samples)'] = df_nonblanks.count(axis=1)

    # subtract 1 from non-blank_samples to account for the compound name
    df_nonblanks['Detection_Count(non-blank_samples)'] = df_nonblanks['Detection_Count(non-blank_samples)'].apply(lambda x: x - 1)

    # total number of samples (subtract 1 for the compound name)
    total_nonblank_samples = len(filter_headers_nonblanks) - 1

    # calculate percentage of samples that have a value and store in new column 'detection_Count(non-blank_samples)(%)'
    df_nonblanks['Detection_Count(non-blank_samples)(%)'] = (df_nonblanks['Detection_Count(non-blank_samples)'] / total_nonblank_samples) * 100
    # round to whole number
    df_nonblanks['Detection_Count(non-blank_samples)(%)'] = df_nonblanks['Detection_Count(non-blank_samples)(%)'].round(0)



    # merge new data into original dataframe
    df_out = pd.merge(df_in, df[[ 'Compound','Detection_Count(all_samples)', 'Detection_Count(all_samples)(%)' ]], how='left', on=['Compound'])
    df_out = pd.merge(df_out, df_nonblanks[[ 'Compound','Detection_Count(non-blank_samples)', 'Detection_Count(non-blank_samples)(%)' ]], how='left', on=['Compound'])
    return df_out


def score(df):  # Get score from annotations.
    """
The Python function score(df) appears to be used to extract a "Score" column from a Pandas DataFrame, df, 
based on the contents of an "Annotations" column within the DataFrame. Here's a breakdown of how the function works:

It defines a regular expression, regex, which is used to extract the score value from the "Annotations" 
column. The regular expression captures the value following "db=" and ending with a comma, closing square bracket, or space.

It first checks if the DataFrame df contains a column named "Annotations" using the condition "Annotations" in df.

If the "Annotations" column exists, it checks whether the entire column is null (contains only NaN values) using 
df.Annotations.isnull().all(). If the entire column is null, meaning there's no useful information in the "Annotations" 
column, it sets the "Score" column in the DataFrame to None for all rows and returns the modified DataFrame.

If the "Annotations" column is not entirely null and contains at least one string that contains "overall=", it 
proceeds to extract the score values from the "Annotations" column.

It uses df.Annotations.str.contains('overall=') to check if any of the strings in the "Annotations" column 
contain the substring "overall=". If such strings are found, it attempts to extract the score values using the 
regular expression defined earlier (df.Annotations.str.extract(regex, expand=True).astype('float64')), and 
stores the extracted scores in a new "Score" column in the DataFrame. The scores are converted to floating-point 
numbers (float64) during extraction.

If an error occurs during the extraction (e.g., if the regular expression doesn't match), it sets the "Score" column to None for all rows.

If the DataFrame does not contain an "Annotations" column and also does not have a "Score" column, it sets the "Score" column in the DataFrame to None for all rows.

Finally, the function returns the modified DataFrame, which may include a new "Score" column based on the extraction process.

This function essentially extracts a "Score" column from the "Annotations" column if certain conditions are met. 
It handles cases where the "Annotations" column is entirely null, where the "Annotations" column contains relevant 
score information, and where the DataFrame doesn't have an "Annotations" column at all.
    """
    regex = "db=(.*?)[, \]].*"  # grab score from first match of db=(value) followed by a , ] or space
    if "Annotations" in df:
        if df.Annotations.isnull().all():  # make sure there isn't a totally blank Annotations column
            df['Score'] = None
            return df
        if df.Annotations.str.contains('overall=').any():
            try:
                df['Score'] = df.Annotations.str.extract(regex, expand=True).astype('float64')
            except ValueError:
                df['Score'] = None
    elif "Score" in df:
        pass
    else:
        df['Score'] = None
    # logging.info("List of scores: {}".format(df['Score']))
    return df


# not yet unit tested
def clean_features(df, controls):  # a method that drops rows based on conditions
    # Abundance=  df.columns[df.columns.str.contains(pat ='N_Abun_')].tolist()
    Abundance=  df.columns[df.columns.str.contains(pat ='Replicate_Percent_')].tolist()
    blanks = ['MB','mb','mB','Mb','blank','Blank','BLANK']
    Mean = df.columns[df.columns.str.contains(pat ='Mean_')].tolist()
    Mean_samples = [md for md in Mean if not any(x in md for x in blanks)]
    Mean_MB = [md for md in Mean if any(x in md for x in blanks)]
    Std = df.columns[df.columns.str.contains(pat ='STD_')].tolist()
    # Std_samples = [md for md in Std if not any(x in md for x in blanks)]
    Std_MB = [md for md in Std if any(x in md for x in blanks)]
    Median = df.columns[df.columns.str.contains(pat ='Median_')].tolist()    
    Median_Samples = [md for md in Median if not any(x in md for x in blanks)]
    Median_Blanks = [md for md in Median if any(x in md for x in blanks)]
    # Median_High = [md for md in Median if 'C' in md]
    # Median_Mid = [md for md in Median if 'B' in md]
    # Median_Low = [md for md in Median if 'A' in md]
    # Median_MB = [md for md in Median if any(x in md for x in blanks)]
    # N_Abun_High = [N for N in Abundance if 'C' in N]
    # N_Abun_MB = [N for N in Abundance if any(x in N for x in blanks)]
    # N_Abun_Samples = [N for N in Abundance if not any(x in N for x in blanks)]
    Replicate_Percent_MB = [N for N in Abundance if any(x in N for x in blanks)]
    Replicate_Percent_Samples = [N for N in Abundance if not any(x in N for x in blanks)]
    #N_Abun_MB= [N for N in Abundanceif 'MB' in N]

    CV = df.columns[df.columns.str.startswith('CV_')].tolist()

    CV_Samples= [C for C in CV if not any(x in C for x in blanks)]
    #set medians where feature abundance is less than some cutoff to nan
    df['AnySamplesDropped'] = np.nan
    for median,N in zip(Median_Samples,Replicate_Percent_Samples):
        #print((str(median) + " , " +str(N)))
        df.loc[df[N] < controls[0], median] = np.nan
        df.loc[df[N] < controls[0], 'AnySamplesDropped'] = 1
    for mean,Std,median,N in zip(Mean_MB,Std_MB,Median_Blanks,Replicate_Percent_MB):
        #print((str(median) + " , " +str(N)))
        df.loc[df[N] < controls[2], median] = np.nan
        df.loc[df[N] < controls[2], mean] = 0
        df.loc[df[N] < controls[2], Std] = 0
    # remove all features where the abundance is less than some cutoff in all samples
    df.drop(df[(df[Replicate_Percent_Samples] < controls[0]).all(axis=1)].index, inplace=True)
    df.drop(df[(df[CV_Samples] > controls[1]).all(axis=1)].index, inplace=True)
    # blank out samples that do not meet the CV cutoff
    cv_not_met = df[CV_Samples] > controls[1]
    m = df[Median_Samples].copy()
    cv_not_met.columns = m.columns
    df[Median_Samples] = m.mask(cv_not_met)
    #find the median of all samples and select features where median_samples/ median_blanks >= cutoff
    #Updated to test sample mean > 3*STDblank + mean_blank
    df['Max_Median_ALLSamples'] = df[Median_Samples].max(axis=1,skipna=True).round(0)

    df['BlkStd_cutoff'] = (3 * df[Std_MB[0]]) + df[Mean_MB[0]]
    df['BlkStd_cutoff'] = df['BlkStd_cutoff'].fillna(df[Mean_MB[0]]) # In the case of a single blank replicate, the previous calculation is an empty value as it cannot calculate Std dev; replace with mean value
    df = df[(df[Replicate_Percent_MB[0]] == 0) | (df[Mean_samples].max(axis=1, skipna=True) > df['BlkStd_cutoff'])]#>=(df['SampletoBlanks_ratio'] >= controls[0])].copy()
    
    return df