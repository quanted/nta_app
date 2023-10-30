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


'''UTILITY FUNCTIONS'''

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
    # changed for NTAW094
    # df.drop_duplicates(subset='Compound',keep='first',inplace=True)
    # formulas = df.loc[df['For_Dashboard_Search'] == '1','Compound'].values
    df.drop_duplicates(subset='Formula',keep='first',inplace=True)
    formulas = df.loc[df['For_Dashboard_Search'] == '1','Formula'].values
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


'''ADDUCT IDENTIFICATION FUNCTIONS'''

# Modified version of Jeff's 'adduct_identifier' function. This is the matrix portion. TMF 10/27/23
def adduct_matrix(df, a_name, delta, Mass_Difference, Retention_Difference, ppm, id_start):
    # 'Mass' to matrix, 'Retention Time' to matrix
    mass = df['Mass'].to_numpy()
    rts = df['Retention_Time'].to_numpy()
    # Reshape 'masses' and 'rts'
    masses_matrix = np.reshape(mass, (len(mass), 1))
    rts_matrix = np.reshape(rts, (len(rts),1))
    # Create difference matrices
    diff_matrix_mass = masses_matrix - masses_matrix.transpose()
    diff_matrix_rt = rts_matrix - rts_matrix.transpose() 
    # Create array of 0s   
    unique_adduct_number = np.zeros(len(df.index))
    # Add 'diff_mass_matrix' by 'delta' (adduct mass)
    is_adduct_diff = abs(diff_matrix_mass - delta)
    has_adduct_diff = abs(diff_matrix_mass + delta)
    # Adjust matrix if units are 'ppm'
    if ppm:
        has_adduct_diff = (has_adduct_diff/masses_matrix)*10**6
        is_adduct_diff = (is_adduct_diff/masses_matrix)*10**6
    # Replace cells in 'has_adduct_diff' below 'Mass_Difference' and 'Retention_Difference' with 1, else 0
    is_adduct_matrix = np.where((is_adduct_diff < Mass_Difference) & (abs(diff_matrix_rt) < Retention_Difference), 1, 0)
    has_adduct_matrix = np.where((has_adduct_diff < Mass_Difference) & (abs(diff_matrix_rt) < Retention_Difference), 1, 0)
    # Remove self matches
    np.fill_diagonal(is_adduct_matrix, 0)
    np.fill_diagonal(has_adduct_matrix, 0)
    # check if all values in is_adduct_matrix are 0
    if np.all(is_adduct_matrix == 0):
        # skip matrix math if no adduct matches 
        pass
    else:
        # Define 'row_num', 'is_id_matrix'
        row_num = len(mass)
        is_id_matrix = np.tile(np.arange(row_num),row_num).reshape((row_num,row_num)) + id_start
        # Matrix multiplication, keep highest # row if multiple adducts
        is_adduct_number = is_adduct_matrix * is_id_matrix
        is_adduct_number_flat = np.max(is_adduct_number, axis=1) # if is adduct of multiple, keep highest # row
        #is_adduct_number_flat_index = np.where(is_adduct_number_flat > 0, is_adduct_number_flat -1, 0)
        #is_adduct_of_adduct = np.where((is_adduct_number_flat > 0) &
        #                               (df['Is_Adduct_or_Loss'][pd.Series(is_adduct_number_flat_index-id_start).clip(lower=0)] > 0), 1, 0)
        #is_adduct_number_flat[is_adduct_of_adduct == 1] = 0
        has_adduct_number = has_adduct_matrix * is_id_matrix
        has_adduct_number_flat = np.max(has_adduct_number, axis=1)  # these will all be the same down columns
        unique_adduct_number = np.where(has_adduct_number_flat != 0, has_adduct_number_flat, is_adduct_number_flat).astype(int)
        # Edit 'df['Has_Adduct_or_Loss']' column
        df['Has_Adduct_or_Loss'] = np.where((has_adduct_number_flat > 0) & (df['Is_Adduct_or_Loss'] == 0),
                                            df['Has_Adduct_or_Loss']+1, df['Has_Adduct_or_Loss'])
        # Edit 'df['Is_Adduct_or_Loss']' column
        df['Is_Adduct_or_Loss'] = np.where((is_adduct_number_flat > 0) & (df['Has_Adduct_or_Loss'] == 0), 1, df['Is_Adduct_or_Loss'])
        # Edit 'df['Adduct_or_Loss_Info']' column
        df['Adduct_or_Loss_Info'] = np.where((has_adduct_number_flat > 0) & (df['Is_Adduct_or_Loss'] == 0),
                                            df['Adduct_or_Loss_Info'] + unique_adduct_number.astype(str) + "({});".format(a_name), df['Adduct_or_Loss_Info'])
        # Edit 'df['Adduct_or_Loss_Info']' column
        df['Adduct_or_Loss_Info'] = np.where((is_adduct_number_flat > 0) & (df['Has_Adduct_or_Loss'] == 0),
                                            df['Adduct_or_Loss_Info'] + unique_adduct_number.astype(str) + "({});".format(a_name), df['Adduct_or_Loss_Info'])
    
    return df


# Estimate a window size from the input data by finding the maximum distance between indices differing by 'mass_diff_mass'. TMF 10/27/23
def window_size(df_in, mass_diff_mass = 112.985586):
    df = df_in.copy()
    masses = df['Mass'].tolist()    
    li=[]
    for i in range(len(masses)-1):
        val = masses[i] + mass_diff_mass
        if df['Mass'].max() > val:
            ind = df.index[df['Mass'] > val].tolist()[0]
            li.append(ind - i)
    window_size = pd.Series(li)
    val = window_size.max()
    
    return val


# Function that takes the input data, chunks it based on window size, then loops through chunks
# and sends them to 'adduct_matrix' for calculation. TMF 10/27/23
def chunk_adducts(df_in, n, step, a_name, delta, Mass_Difference, Retention_Difference, ppm, id_start):
    # Create copy
    df=df_in.copy()
    # Create chunks of df based on how much Web App can handle (n) and step size that captures all adducts (step)
    to_test_list = [df[i:i+n] for i in range(0, df.shape[0], step)]
    to_test_list = [i for i in to_test_list if (i.shape[0] > n/2)]
    # Create list, iterate through df chunks and append results to list   
    li=[]
    for x in to_test_list:
        dum = adduct_matrix(x, a_name, delta, Mass_Difference, Retention_Difference, ppm, id_start)
        li.append(dum)
    # Concatenate results together, removing overlapping sections
    output = pd.concat(li, axis=0).drop_duplicates(subset = ['Mass', 'Retention_Time'], keep = 'last')
    
    return output


# Function that does the front-end of the old 'adduct_identifier'; we trim the input data by identifying
# features that are near to adduct distance from another feature. This shortened dataframe is used to 
# calculate a window size, then loop through possible adducts, passing to 'chunk_adducts'. TMF 10/27/23
def adduct_identifier(df_in, Mass_Difference, Retention_Difference, ppm, ionization, id_start = 0):
    # Copy df_in, only need 'Mass' and 'Retention Time'
    df = df_in[['Mass', 'Retention_Time']].copy()
    # Round columns
    df['Rounded Mass'] = df['Mass'].round(2)
    df['Rounded RT'] = df['Retention_Time'].round(1)
    # Create tuple of 'Rounded RT' and 'Rounded Mass'
    df['Rounded_RT_Mass_Pair'] = list(zip(df['Rounded RT'], df['Rounded Mass'])) 
    # Define pos/neg/neutral adduct dictionaries, proton
    pos_adduct_deltas = {'Na': 22.989218, 'K': 38.963158, 'NH4': 18.033823}
    neg_adduct_deltas = {'Cl': 34.969402, 'Br': 78.918885, 'HCO2': 44.998201, 'CH3CO2': 59.013851, 'CF3CO2': 112.985586}
    neutral_loss_deltas= {'H2O': -18.010565, 'CO2': -43.989829}
    proton_mass = 1.007276
    # Determine possible adduct dictionary according to ionization
    if ionization == "positive":
        # we observe Mass+(H+) and Mass+(Adduct)
        possible_adduct_deltas = {k: v - proton_mass for (k,v) in pos_adduct_deltas.items()}
    else:
        # we observe Mass-(H+) and Mass+(Adduct)
        possible_adduct_deltas = {k: v + proton_mass for (k,v) in neg_adduct_deltas.items()}
        # add neutral loss adducts
        possible_adduct_deltas.update(neutral_loss_deltas)
    # Create empty list to hold mass shift/RT tuples
    list_of_mass_shifts_RT_pairs = []
    # Loop through possible adducts, add/subtract adduct mass from each feature, append
    # 'Rounded RT', 'Rounded Mass' tuples to 'list_of_mass_shifts_RT_pairs' for both addition
    # and subtraction.
    for (k,v) in possible_adduct_deltas.items():
        col1 = 'Mass - '+k
        col2 = 'Mass + '+k
        df[col1] = (df['Mass'] - v).round(2)
        df[col2] = (df['Mass'] + v).round(2)
        list_of_mass_shifts_RT_pairs.append(list(zip(df['Rounded RT'], df[col1])))
        list_of_mass_shifts_RT_pairs.append(list(zip(df['Rounded RT'], df[col2])))
    # Extend list  
    list_of_mass_shifts_RT_pairs = [item for sublist in list_of_mass_shifts_RT_pairs for item in sublist]
    # Remove duplicate tuples (sets don't carry duplicates)
    list_of_mass_shifts_RT_pairs = list(set(list_of_mass_shifts_RT_pairs))
    # Filter df for features to check for adducts
    to_test = df[df['Rounded_RT_Mass_Pair'].isin(list_of_mass_shifts_RT_pairs)]
    to_test = to_test.sort_values('Mass', ignore_index=True) 
    # Add columns to be changed by 'adduct_matrix'
    to_test['Has_Adduct_or_Loss'] = 0
    to_test['Is_Adduct_or_Loss'] = 0
    to_test['Adduct_or_Loss_Info'] = ""
    # Set 'n' to tested memory capacity of WebApp for number of features in 'adduct_matrix'
    n = 12000
    # If 'to_test' is less than n, send it straight to 'adduct_matrix'
    if to_test.shape[0] <= n:
        for a_name, delta in possible_adduct_deltas.items():
            to_test = adduct_matrix(to_test, a_name, delta, Mass_Difference, Retention_Difference, ppm, id_start)
    # Else, calculate the moving window size and send 'to_test' to 'chunk_adducts'
    else:  
        step = n - window_size(to_test)
        # Loop through possible adducts, perform 'adduct_matrix'
        for a_name, delta in possible_adduct_deltas.items():
            to_test = chunk_adducts(to_test, n, step, a_name, delta, Mass_Difference, Retention_Difference, ppm, id_start)
    # Concatenate 'Has_Adduct_or_Loss', 'Is_Adduct_or_Loss', 'Adduct_or_Loss_Info' to df
    df_in = pd.merge(df_in, to_test[['Mass', 'Retention_Time', 'Has_Adduct_or_Loss','Is_Adduct_or_Loss','Adduct_or_Loss_Info']],
                  how = 'left', on = ['Mass', 'Retention_Time'])
    
    return df_in


'''DUPLICATE REMOVAL FUNCTIONS'''

# Wrapper function for passing manageable-sized dataframe chunks to 'dup_matrix'. TMF 10/27/23
def chunk_duplicates(df_in, n, step, mass_cutoff, rt_cutoff):
    # Create copy of df_in
    df=df_in.copy()
    # Chunk df based on n (# of features WebApp can handle) and step
    to_test_list = [df[i:i+n] for i in range(0, df.shape[0], step)]
    # Remove small remainder tail, if present
    to_test_list = [i for i in to_test_list if (i.shape[0] > n/2)]
        
    li=[]
    dupe_li = []
    # Pass list to 'dup_matrix'
    for x in to_test_list:
        #dum = dup_matrix(x, mass_cutoff, rt_cutoff) # Deprecated 10/30/23 -- TMF
        dum, dupes = dup_matrix(x, mass_cutoff, rt_cutoff)
        li.append(dum)
        dupe_li.append(dupes)
    # Concatenate results, drop duplicates from overlap
    output = pd.concat(li, axis=0).drop_duplicates(subset = ['Mass', 'Retention_Time'], keep = 'first')
    dupe_df = pd.concat(dupe_li, axis=0).drop_duplicates(subset = ['Mass', 'Retention_Time'], keep = 'first')
    
    return output, dupe_df


# Called within the 'duplicates' function - takes a filtered 'to_test' df, does matrix math, returns 'passed'. TMF 10/27/23
def dup_matrix(df_in, mass_cutoff, rt_cutoff):
    # Create matrices from df_in
    mass = df_in['Mass'].to_numpy()
    rts = df_in['Retention_Time'].to_numpy()
    # Reshape matrices
    masses_matrix = np.reshape(mass, (len(mass), 1))
    rts_matrix = np.reshape(rts, (len(rts), 1))
    # Perform matrix transposition
    diff_matrix_mass = masses_matrix - masses_matrix.transpose()
    diff_matrix_rt = rts_matrix - rts_matrix.transpose()
    # Find indices where differences are less than 'mass_cutoff' and 'rt_cutoff'
    duplicates_matrix = np.where((abs(diff_matrix_mass) <= mass_cutoff) & (abs(diff_matrix_rt) <= rt_cutoff),1,0)
    np.fill_diagonal(duplicates_matrix, 0)
    # Find # of duplicates for each row
    row_sums = np.sum(duplicates_matrix, axis=1)
    # Calculate lower triangle of matrix
    duplicates_matrix_lower = np.tril(duplicates_matrix)
    lower_row_sums = np.sum(duplicates_matrix_lower, axis=1)
    # Store features with no duplicates in 'passed'
    passed = df_in[(row_sums == 0) | (lower_row_sums == 0)].copy()
    # Flag duplicates as 'D'
    dupes = df_in.loc[df_in[(row_sums!=0) & (lower_row_sums != 0)].index,:]
    
    return passed, dupes


# Drop duplicates from input dataframe, based on mass_cutoff and rt_cutoff. TMF 10/27/23    
def duplicates(df_in, mass_cutoff=0.005, rt_cutoff=0.25):
    # Copy the dataframe
    df = df_in.copy()
    # Parse headers to find sample columns
    all_headers = parse_headers(df) 
    sam_headers = [item for sublist in all_headers for item in sublist if len(sublist) > 1]
    # Calculate 'all_sample_mean', sort df by 'all_sample_mean', reset index
    df['all_sample_mean'] = df[sam_headers].mean(axis=1)  # mean intensity across all samples
    df.sort_values(by=['all_sample_mean'], inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)
    # Define feature limit of WebApp
    n=12000
    step=6000
    # 'if' statement for chunker: if no chunks needed, send to 'dup_matrix', else send to 'chunk_duplicates'
    if df.shape[0] <= n:
        #output = dup_matrix(df, mass_cutoff, rt_cutoff) # Deprecated 10/30/23 -- TMF
        output, dupe_df = dup_matrix(df, mass_cutoff, rt_cutoff) 
    else:
        #output = chunk_duplicates(df, n, step, mass_cutoff, rt_cutoff) # Deprecated 10/30/23 -- TMF
        output, dupe_df = chunk_duplicates(df, n, step, mass_cutoff, rt_cutoff)
    # Sort output by 'Mass', reset the index, drop 'all_sample_mean'
    output.sort_values(by=['Mass'], inplace=True)
    output.reset_index(drop=True, inplace=True)
    output.drop(['all_sample_mean'], axis=1, inplace=True)
    dupe_df.drop(['all_sample_mean'], axis=1, inplace=True)
    
    return output, dupe_df


'''CALCULATE STATISTICS FUNCTIONS'''
# TMF 10/27/23
def statistics(df_in):
    # Create copy
    df = df_in.copy()
    # Parse headers, get sample headers
    all_headers = parse_headers(df_in) 
    sam_headers = [i for i in all_headers if len(i) > 1]
    # Create column names for each statistics from sam_headers
    mean_cols = ['Mean_' + i[0][:-1] for i in sam_headers]
    med_cols = ['Median_' + i[0][:-1] for i in sam_headers]
    std_cols = ['STD_' + i[0][:-1] for i in sam_headers]
    cv_cols = ['CV_' + i[0][:-1] for i in sam_headers]
    nabun_cols = ['N_Abun_' + i[0][:-1] for i in sam_headers]
    rper_cols = ['Replicate_Percent_' + i[0][:-1] for i in sam_headers]
    # Concatenate list comprehensions to calculate each statistic for each sample
    means = pd.concat([df[x].mean(axis=1).round(4).rename(col) for x, col in zip(sam_headers, mean_cols)], axis=1)
    medians = pd.concat([df[x].median(axis=1, skipna=True).round(4).rename(col) for x, col in zip(sam_headers, med_cols)], axis=1)
    stds = pd.concat([df[x].std(axis=1, skipna=True).round(4).rename(col) for x, col in zip(sam_headers, std_cols)], axis=1)
    cvs = pd.concat([(stds[scol]/means[mcol]).round(4).rename(col) for mcol, scol, col in zip(mean_cols, std_cols, cv_cols)], axis=1)
    nabuns = pd.concat([df[x].count(axis=1).round(0).rename(col) for x, col in zip(sam_headers, nabun_cols)], axis=1)
    rpers = pd.concat([((nabuns[ncol]/len(x)).round(4)*100).rename(col) for x, ncol, col in zip(sam_headers, nabun_cols, rper_cols)], axis=1)
    # Concatenate all statistics together
    output = pd.concat([df, means, medians, stds, cvs, nabuns, rpers], axis=1)
    
    return output
   
    
# TMF 10/27/23    
def chunk_stats(df_in):
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
    # iterate through list_df, calculating 'statistics' on chunks and appending to li
    for df in list_df:
        li.append(statistics(df))
    # concatenate li, sort, and calculate 'Rounded_Mass' + 'Max_CV_across_sample'
    output = pd.concat(li, axis=0)
    output.sort_values(['Mass', 'Retention_Time'], ascending=[True, True], inplace=True)
    output['Rounded_Mass'] = output['Mass'].round(0)
    output['Max_CV_across_sample'] = output.filter(regex='CV_').max(axis=1)
    
    return output
    

'''FUNCTION FOR CLEANING FEATURES'''

''' NEW FUNCTION FOR CLEANING FEATURES THAT ALSO DOCUMENTS FLAGS IN A SEPARATE DATAFRAME - DETECTION COUNT IS MIGRATING TO THIS FUNCTION
    TMF 10/27/23 '''
    
def clean_features(df_in, controls):  # a method that drops rows based on conditions
    # Make dataframe copy, create docs in df's image
    df = df_in.copy()
    docs = pd.DataFrame().reindex_like(df)
    docs['Mass'] = df['Mass']
    docs['Retention_Time'] = df['Retention_Time']   
    # Define lists
    blanks = ['MB','mb','mB','Mb','blank','Blank','BLANK']
    Abundance=  df.columns[df.columns.str.contains(pat ='Replicate_Percent_')].tolist()
    Replicate_Percent_MB = [N for N in Abundance if any(x in N for x in blanks)]
    Replicate_Percent_Samples = [N for N in Abundance if not any(x in N for x in blanks)]
    Mean = df.columns[df.columns.str.contains(pat ='Mean_')].tolist()
    Mean_Samples = [md for md in Mean if not any(x in md for x in blanks)]
    Mean_MB = [md for md in Mean if any(x in md for x in blanks)]   
    Std = df.columns[df.columns.str.contains(pat ='STD_')].tolist()
    Std_MB = [md for md in Std if any(x in md for x in blanks)]   
    Median = df.columns[df.columns.str.contains(pat ='Median_')].tolist()    
    Median_Samples = [md for md in Median if not any(x in md for x in blanks)]
    Median_Blanks = [md for md in Median if any(x in md for x in blanks)]
    CV = df.columns[df.columns.str.startswith('CV_')].tolist()
    CV_Samples= [C for C in CV if not any(x in C for x in blanks)]
    
    ## REPLICATE FLAG
    # Set medians where feature presence is less than some replicate percentage cutoff to nan
    df['AnySamplesDropped'] = np.nan
    for median,N in zip(Median_Samples,Replicate_Percent_Samples):
        df.loc[df[N] < controls[0], median] = np.nan
        docs.loc[df[N] < controls[0], median] = 'R'
        df.loc[df[N] < controls[0], 'AnySamplesDropped'] = 1
    for mean,Std,median,N in zip(Mean_MB,Std_MB,Median_Blanks,Replicate_Percent_MB):
        df.loc[df[N] < controls[2], median] = np.nan
        df.loc[df[N] < controls[2], mean] = 0
        df.loc[df[N] < controls[2], Std] = 0
        docs.loc[df[N] < controls[2], median] = 'R'
        docs.loc[df[N] < controls[2], mean] = 'R'
        docs.loc[df[N] < controls[2], Std] = 'R'  
    # update docs with 'AnySamplesDropped' column
    docs['AnySamplesDropped'] = df['AnySamplesDropped']
    
    ## CV FLAG
    # Create a mask for df based on sample-level CV threshold 
    cv_not_met = df[CV_Samples] > controls[1]
    m = df[Median_Samples].copy()
    cv_not_met.columns = m.columns
    # Blank out sample medians where occurrence does not meet CV cutoff
    df[Median_Samples] = m.mask(cv_not_met)     
    # Create empty cell mask from documentation dataframe
    cell_empty = docs[Median_Samples].isnull()    
    # append CV flag (CV > threshold) to documentation dataframe
    docs[Median_Samples] = np.where(cv_not_met & cell_empty, 'CV', docs[Median_Samples])
    docs[Median_Samples] = np.where(cv_not_met & ~cell_empty, docs[Median_Samples]+', CV', docs[Median_Samples])
    
    ## MDL CALCULATION/MASKS
    # Calculate feature MDL
    df['BlkStd_cutoff'] = (3 * df[Std_MB[0]]) + df[Mean_MB[0]]
    df['BlkStd_cutoff'] = df['BlkStd_cutoff'].fillna(df[Mean_MB[0]]) # In the case of a single blank replicate, the previous calculation is an empty value as it cannot calculate Std dev; replace with mean value
    df['BlkStd_cutoff'] = df['BlkStd_cutoff'].fillna(0)
    docs['BlkStd_cutoff'] = df['BlkStd_cutoff']  
    # Create a mask for docs based on sample-level MDL threshold 
    # Median Masks
    MDL_all_mask = pd.DataFrame().reindex_like(df[Median])  
    for x,y in zip(Median, Mean):
        MDL_all_mask[x] = df[y] > df['BlkStd_cutoff']   
    MDL_sample_mask = pd.DataFrame().reindex_like(df[Median_Samples])  
    for x,y in zip(Median_Samples, Mean_Samples):
        MDL_sample_mask[x] = df[y] > df['BlkStd_cutoff']  

    ## CALCULATE DETECTION COUNTS
    # Calculate Detection_Count
    df['Detection_Count(all_samples)'] = MDL_all_mask.sum(axis=1)
    df['Detection_Count(non-blank_samples)'] = MDL_sample_mask.sum(axis=1)
    # total number of samples (subtract 1 for the compound name)
    med_total = len(Median)
    med_samples = len(Median_Samples)
    # calculate percentage of samples that have a value and store in new column 'detection_Count(all_samples)(%)'
    df['Detection_Count(all_samples)(%)'] = (df['Detection_Count(all_samples)'] / med_total) * 100
    df['Detection_Count(all_samples)(%)'] = df['Detection_Count(all_samples)(%)'].round(1)
    # calculate percentage of samples that have a value and store in new column 'detection_Count(non-blank_samples)(%)'
    df['Detection_Count(non-blank_samples)(%)'] = (df['Detection_Count(non-blank_samples)'] / med_samples) * 100
    df['Detection_Count(non-blank_samples)(%)'] = df['Detection_Count(non-blank_samples)(%)'].round(1)
    # Assign to docs
    docs['Detection_Count(all_samples)'] = df['Detection_Count(all_samples)']
    docs['Detection_Count(non-blank_samples)'] = df['Detection_Count(non-blank_samples)']
    docs['Detection_Count(all_samples)(%)'] = df['Detection_Count(all_samples)(%)']
    docs['Detection_Count(non-blank_samples)(%)'] = df['Detection_Count(non-blank_samples)(%)']
        
    ## MDL/ND FLAG
    # Create updated empty cell mask from documentation dataframe
    cell_empty = docs[Median_Samples].isnull()
    # append ND flag (occurrence < MDL) to documentation dataframe
    docs[Median_Samples] = np.where(~MDL_sample_mask & cell_empty, 'ND', docs[Median_Samples])
    docs[Median_Samples] = np.where(~MDL_sample_mask & ~cell_empty, docs[Median_Samples]+', ND', docs[Median_Samples])
    
    ## ADD VALUES TO DOC
    # Mask, add values back to doc
    values = docs[Median_Samples].isnull()
    docs[Median_Samples] = np.where(values, df[Median_Samples], docs[Median_Samples])
    
    ## DROP FEATURES FROM DF
    # remove all features where the abundance is less than some cutoff in all samples
    df.drop(df[(df[Replicate_Percent_Samples] < controls[0]).all(axis=1)].index, inplace=True)
    df.drop(df[(df[CV_Samples] > controls[1]).all(axis=1)].index, inplace=True) 
    # Keep samples where the feature doesn't exist in the blank OR at least one sample mean exceeds MDL
    df = df[(df[Replicate_Percent_MB[0]] == 0) | (df[Mean_Samples].max(axis=1, skipna=True) > df['BlkStd_cutoff'])]#>=(df['SampletoBlanks_ratio'] >= controls[0])].copy()
    
    return df, docs


'''FUNCTIONS FOR COMBINING DATAFRAMES'''

# Combine function for the self.dfs
def combine(df1,df2):
    if df1 is not None and df2 is not None:
        dfc = pd.concat([df1,df2], sort=True) #fixing pandas FutureWarning
        dfc = dfc.reindex(columns = df1.columns)
    elif df1 is not None:
        dfc = df1.copy()
    else:
        dfc = df2.copy()
    columns = dfc.columns.values.tolist()

    # create new flags
    # NTAW-94
    # dfc = dfc.drop_duplicates(subset=['Compound','Mass','Retention_Time','Score'])
    # dfc['N_Compound_Hits'] = dfc.groupby('Compound')['Compound'].transform('size')
    dfc = dfc.drop_duplicates(subset=['Mass','Retention_Time'])
    # dfc['N_Compound_Hits'] = dfc.groupby('Compound')['Compound'].transform('size')

    Median_list =  dfc.columns[(dfc.columns.str.contains(pat ='Median_')==True)\
                 & (dfc.columns.str.contains(pat ='MB|blank|blanks|BlankSub|_x|_y')==False)].tolist()
    #print(Median_list)
    dfc['N_Abun_Samples'] = dfc[Median_list].count(axis=1,numeric_only=True)
    dfc['Median_Abun_Samples'] = dfc[Median_list].median(axis=1,skipna=True).round(0)

    # NTAW-94
    # dfc = dfc[columns].sort_values(['Compound'],ascending=[True])
    dfc = dfc[columns].sort_values(['Mass','Retention_Time'],ascending=[True,True])
    return dfc
    

# Combine function for the self.docs and self.dupes
def combine_doc(doc,dupe):
    
    Median = doc.columns[doc.columns.str.contains(pat ='Median_')].tolist()
    # Median = doc.columns[doc.columns.str.contains(pat = 'BlankSub_')].tolist()
    
    if doc is not None and dupe is not None:
        dupe.loc[:, Median] = 'D'
        dfc = pd.concat([doc,dupe], sort=True) #fixing pandas FutureWarning
        dfc = dfc.reindex(columns = doc.columns)
    elif doc is not None:
        dfc = doc.copy()
    else:
        dfc = dupe.copy()
  
    to_keep = ['Compound', 'Mass', 'Retention_Time', 'BlkStd_cutoff'] + Median 
    dfc = dfc[to_keep]
    
    return dfc



'''
def cal_detection_count(df_in):
    blanks = ['MB','mb','mB','Mb','blank','Blank','BLANK']

    # make a working copy of the dataframe
    df = df_in.copy()

    # a list of lists of headers that contain abundance data
    all_header_groups = parse_headers(df)
    abundance = [item for sublist in all_header_groups for item in sublist if len(sublist) > 1]
    # NTAW-94 remove 'Compound' from list of abundance
    # filter_headers= ['Compound'] + abundance
    filter_headers= ['Mass', "Retention_Time"] + abundance

    # remove all items filter_headers containing string 'BlankSub_Median_' from list filter_headers
    filter_headers = [item for item in filter_headers if 'BlankSub_Median_' not in item]

    filter_headers_nonblanks = [item for item in filter_headers if not any(x in item for x in blanks)]

# Std_samples = [md for md in Std if not any(x in md for x in blanks)]

    df = df[filter_headers].copy()
    df_nonblanks = df[filter_headers_nonblanks].copy()

    # calculate detection_Count
    df['Detection_Count(all_samples)'] = df.count(axis=1)

    # subtract 2 from detection_Count to account for the 'Mass', "Retention_Time"
    df['Detection_Count(all_samples)'] = df['Detection_Count(all_samples)'].apply(lambda x: x - 2)

    # total number of samples (subtract 2 for the 'Mass', "Retention_Time")
    total_samples = len(filter_headers) - 2

    # calculate percentage of samples that have a value and store in new column 'detection_Count(all_samples)(%)'
    df['Detection_Count(all_samples)(%)'] = (df['Detection_Count(all_samples)'] / total_samples) * 100
    # round to whole number
    df['Detection_Count(all_samples)(%)'] = df['Detection_Count(all_samples)(%)'].round(0)




    # calculate non-blank_samples
    df_nonblanks['Detection_Count(non-blank_samples)'] = df_nonblanks.count(axis=1)

    # subtract 2 from non-blank_samples to account for the 'Mass', "Retention_Time"
    df_nonblanks['Detection_Count(non-blank_samples)'] = df_nonblanks['Detection_Count(non-blank_samples)'].apply(lambda x: x - 2)

    # total number of samples (subtract 2 for the 'Mass', "Retention_Time")
    total_nonblank_samples = len(filter_headers_nonblanks) - 2

    # calculate percentage of samples that have a value and store in new column 'detection_Count(non-blank_samples)(%)'
    df_nonblanks['Detection_Count(non-blank_samples)(%)'] = (df_nonblanks['Detection_Count(non-blank_samples)'] / total_nonblank_samples) * 100
    # round to whole number
    df_nonblanks['Detection_Count(non-blank_samples)(%)'] = df_nonblanks['Detection_Count(non-blank_samples)(%)'].round(0)



    # merge new data into original dataframe
    # NYAW-94
    # df_out = pd.merge(df_in, df[[ 'Compound','Detection_Count(all_samples)', 'Detection_Count(all_samples)(%)' ]], how='left', on=['Compound'])
    # df_out = pd.merge(df_out, df_nonblanks[[ 'Compound','Detection_Count(non-blank_samples)', 'Detection_Count(non-blank_samples)(%)' ]], how='left', on=['Compound'])
    df_out = pd.merge(df_in, df[[ 'Mass', "Retention_Time",'Detection_Count(all_samples)', 'Detection_Count(all_samples)(%)' ]], how='left', on=['Mass', "Retention_Time"])
    df_out = pd.merge(df_out, df_nonblanks[[ 'Mass', "Retention_Time",'Detection_Count(non-blank_samples)', 'Detection_Count(non-blank_samples)(%)' ]], how='left', on=['Mass', "Retention_Time"])
    return df_out
'''

'''
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
  '''  
  
  
  
  
  
'''OLD VERSIONS OF FUNCTIONS -- NEED TO MOVE OR DELETE''' 
 
 
'''
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
        is_id_matrix = np.tile(np.arange(row_num),row_num).reshape((row_num,row_num)) + id_start #don't need to do the reshape if we pass a tuple: np.tile(np.arange(row_num), (row_num, row_num))
        #has_id_matrix = is_id_matrix.transpose()
        is_adduct_number = is_adduct_matrix * is_id_matrix
        is_adduct_number_flat = np.max(is_adduct_number, axis=1) # if is adduct of multiple, keep highest # row; unlikely to happen?
        is_adduct_number_flat_index = np.where(is_adduct_number_flat > 0, is_adduct_number_flat -1, 0) # should this be '- id_start' ? --> maybe get rid of '- 1', but need to test this
        is_adduct_of_adduct = np.where((is_adduct_number_flat > 0) &
                                       (df['Is_Adduct_or_Loss'][pd.Series(is_adduct_number_flat_index-id_start).clip(lower=0)] > 0), 1, 0)
        is_adduct_number_flat[is_adduct_of_adduct == 1] = 0
        has_adduct_number = has_adduct_matrix * is_id_matrix
        has_adduct_number_flat = np.max(has_adduct_number, axis=1)  # these will all be the same down columns
        unique_adduct_number = np.where(has_adduct_number_flat != 0, has_adduct_number_flat, is_adduct_number_flat).astype(int) # preferentially stores having an adduct over being an adduct
        #unique_adduct_number = np.where(unique_adduct_number == 0, unique_adduct_number_new, unique_adduct_number)
        df['Has_Adduct_or_Loss'] = np.where((has_adduct_number_flat > 0) & (df['Is_Adduct_or_Loss'] == 0),
                                            df['Has_Adduct_or_Loss']+1, df['Has_Adduct_or_Loss'])                # this is labeled as a binary, but is actually a count!
        df['Is_Adduct_or_Loss'] = np.where((is_adduct_number_flat > 0) & (df['Has_Adduct_or_Loss'] == 0), 1, df['Is_Adduct_or_Loss'])
        # new_cols = ['unique_{}_number'.format(a_name), 'has_{}_adduct'.format(a_name), 'is_{}_adduct'.format(a_name)]
        #unique_adduct_number_str =
        logger.info("Checkpoint xi")
        df['Adduct_or_Loss_Info'] = np.where((has_adduct_number_flat > 0) & (df['Is_Adduct_or_Loss'] == 0),
                                             df['Adduct_or_Loss_Info'] + unique_adduct_number.astype(str) + "({});".format(a_name), df['Adduct_or_Loss_Info'])
        df['Adduct_or_Loss_Info'] = np.where((is_adduct_number_flat > 0) & (df['Has_Adduct_or_Loss'] == 0),
                                             df['Adduct_or_Loss_Info'] + unique_adduct_number.astype(str) + "({});".format(a_name), df['Adduct_or_Loss_Info'])
    return df
'''


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
   