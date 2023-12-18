import pandas as pd
import numpy as np
from operator import itemgetter
from difflib import SequenceMatcher
from itertools import groupby
import os
import re
import logging


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
    df.drop_duplicates(subset='Formula',keep='first',inplace=True)
    formulas = df.loc[df['For_Dashboard_Search'] == '1','Formula'].values
    formulas_list = [str(i) for i in formulas]
    return formulas_list


def masses(df):
    masses = df.loc[df['For_Dashboard_Search'] == '1','Mass'].values
    logger.info('# of masses for dashboard search: {} out of {}'.format(len(masses),len(df)))
    masses_list = [str(i) for i in masses]
    return masses_list


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


'''PASS THROUGH COLUMNS'''

# Find all columns in dfs that aren't necessary and store for the output
def passthrucol(df_in):
    # Make a copy of the input df
    df=df_in.copy()
    # Parse headers
    all_headers = parse_headers(df)
    # Define active_cols: Keep 'Feature_ID' in pt_headers to merge later
    active_cols = ['Retention_Time', 'Mass', 'Ionization_Mode', 'Formula']
    # Create list of pass through headers that are not in the active columns
    pt_headers = ['Feature_ID'] + [item for sublist in all_headers for item in sublist if len(sublist) == 1 and not any(x in sublist for x in active_cols)]
    headers = ['Feature_ID'] + [item for sublist in all_headers for item in sublist if not any(x in item for x in pt_headers)]
    # Save pass through columns in df
    df_pt = df[pt_headers]
    df_trim = df[headers]
    
    return df_pt, df_trim


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
    # Set chunk size (i.e., # rows)
    n = 5000
    # 'if' statement for chunks: if no chunks needed, send to 'statistics', else chunk and iterate
    if df.shape[0] < n:
        output = statistics(df)
    else:
        # Create list of Data.Frame chunks
        list_df = [df[i:i+n] for i in range(0,df.shape[0],n)]
        # Instantiate empty list
        li=[]
        # iterate through list_df, calculating 'statistics' on chunks and appending to li
        for df in list_df:
            li.append(statistics(df))
        # concatenate li, sort, and calculate 'Rounded_Mass' + 'Max_CV_across_sample'
        output = pd.concat(li, axis=0)
    # Sort output mass and add two new columns
    output.sort_values(['Mass', 'Retention_Time'], ascending=[True, True], inplace=True)
    output['Rounded_Mass'] = output['Mass'].round(0)
    output['Max_CV_across_sample'] = output.filter(regex='CV_').max(axis=1)
    # Define lists to calculate MRL for inclusion in 'Feature_statistics' outputs
    blanks = ['MB','mb','mB','Mb','blank','Blank','BLANK']
    Mean = output.columns[output.columns.str.contains(pat ='Mean_')].tolist()
    Mean_MB = [md for md in Mean if any(x in md for x in blanks)]   
    Std = output.columns[output.columns.str.contains(pat ='STD_')].tolist()
    Std_MB = [md for md in Std if any(x in md for x in blanks)]
    # Calculate feature MRL
    output['MRL'] = (3 * output[Std_MB[0]]) + output[Mean_MB[0]]
    output['MRL'] = output['MRL'].fillna(output[Mean_MB[0]])
    output['MRL'] = output['MRL'].fillna(0)
    
    return output


# Sort columns for the data feature statistics outputs; TMF 11/21/23
def column_sort_DFS(df_in):
    df = df_in.copy()
    # Parse headers
    all_headers = parse_headers(df)
    # Get all cols, group roots (i.e., drop unique value from sample groups)
    all_cols = df.columns.tolist()
    group_cols = [sublist[0][:-1] for sublist in all_headers if len(sublist) > 1]
    # Create list of prefixes to remove non-samples
    prefixes = ['Mean_','Median_', 'CV_', 'STD_', 'N_Abun_', 'Replicate_Percent_', 'Detection']   
    # Isolate sample_groups from prefixes columns   
    groups = [item for item in group_cols if not any(x in item for x in prefixes)]
    # Organize front matter
    front_matter = [item for item in all_cols if not any(x in item for x in groups)]
    ids = ['Feature_ID', 'Mass', 'Retention_Time', 'Ionization_Mode']
    #ids = ['Compound Name', 'Mass', 'Retention_Time', 'Ionization_Mode']
    front_matter = [item for item in front_matter if not any(x in item for x in ids)]
    front_matter = ids + front_matter
    # Organize stats columns
    cols = []
    for sam in groups:
        group_stats = [item for item in all_cols if sam in item]
        cols.append(group_stats)    
    stats_cols = sum(cols, [])
    # Combine into new column list
    new_col_org = front_matter + stats_cols
    # Subset data with new column list
    df_reorg = df[new_col_org]
    
    return df_reorg


# Sort columns for tracer sample results
def column_sort_TSR(df_in):
    df = df_in.copy()
    all_cols = df.columns.tolist()
    # Create list of prefixes to remove non-samples
    prefixes = ['Feature_ID', 'Mass', 'Retention_Time']   
    # Isolate sample_groups from prefixes columns   
    back_matter = [item for item in all_cols if not any(x in item for x in prefixes)]
    # Organize front matter
    front_matter = ['Feature_ID', 'Observed_Mass', 'Observed_Retention_Time',
                    'Monoisotopic_Mass', 'Retention_Time',
                    'Mass_Error_PPM', 'Retention_Time_Difference']
    # Combine into new column list
    new_col_org = front_matter + back_matter
    # Subset data with new column list
    df_reorg = df[new_col_org]
    
    return df_reorg



'''FUNCTION FOR CHECKING TRACERS'''

''' UPDATED FUNCTION FOR CHECKING TRACERS THAT APPENDS 'Tracer_chemical_match' TO THE DFS AND CALCULATES OCCURRENCE COUNT
    TMF 12/11/23 '''

def check_feature_tracers(df,tracers_file,Mass_Difference,Retention_Difference,ppm):
    df1 = df.copy()
    df2 = tracers_file.copy()
    # Get sample names
    prefixes = ['Mean_','Median_', 'CV_', 'STD_', 'N_Abun_', 'Replicate_Percent_', 'Detection']  
    all_headers = parse_headers(df1)
    samples = [item for subgroup in all_headers for item in subgroup if ((len(subgroup) > 1) and not any(x in item for x in prefixes))]
    # Replace all caps or all lowercase ionization mode with "Esi" in order to match correctly to sample data dataframe
    df2['Ionization_Mode'] = df2['Ionization_Mode'].replace('ESI+','Esi+')
    df2['Ionization_Mode'] = df2['Ionization_Mode'].replace('esi+','Esi+')
    df2['Ionization_Mode'] = df2['Ionization_Mode'].replace('ESI-','Esi-')
    df2['Ionization_Mode'] = df2['Ionization_Mode'].replace('esi-','Esi-')
    # Create 'Rounded_Mass' variable to merge on
    df2['Rounded_Mass'] = df2['Monoisotopic_Mass'].round(0)
    df1.rename(columns = {'Mass':'Observed_Mass','Retention_Time':'Observed_Retention_Time'},inplace=True)
    df1['Rounded_Mass'] = df1['Observed_Mass'].round(0)
    # Merge df and tracers
    dft = pd.merge(df2,df1,how='left',on=['Rounded_Mass','Ionization_Mode'])
    if ppm:
        dft['Matches'] = np.where((abs((dft['Monoisotopic_Mass']-dft['Observed_Mass'])/dft['Monoisotopic_Mass'])*1000000<=Mass_Difference) & (abs(dft['Retention_Time']-dft['Observed_Retention_Time'])<=Retention_Difference) ,1,0)
    else:
        dft['Matches'] = np.where((abs(dft['Monoisotopic_Mass']-dft['Observed_Mass'])<=Mass_Difference) & (abs(dft['Retention_Time']-dft['Observed_Retention_Time'])<=Retention_Difference) ,1,0)
    dft = dft[dft['Matches']==1]
    # Caculate Occurrence Count and % in tracers
    dft['Occurrence_Count(across_all_replicates)'] = dft[samples].count(axis=1)
    dft['Occurrence_Count(across_all_replicates)(%)'] = (dft['Occurrence_Count(across_all_replicates)'] / len(samples)) * 100
    # Get 'Matches' info into main df
    dum = dft[['Observed_Mass', 'Observed_Retention_Time', 'Matches']].copy()
    dfc = pd.merge(df1, dum, how='left', on=['Observed_Mass', 'Observed_Retention_Time'])
    dfc.rename(columns = {'Observed_Mass':'Mass','Observed_Retention_Time':'Retention_Time', 'Matches':'Tracer_chemical_match'},inplace=True)
    # Drop columns
    dft.drop(['Rounded_Mass','Matches'],axis=1,inplace=True)
    
    return dft, dfc



'''FUNCTION FOR CLEANING FEATURES'''

''' NEW FUNCTION FOR CLEANING FEATURES THAT ALSO DOCUMENTS FLAGS IN A SEPARATE DATAFRAME - DETECTION COUNT IS MIGRATING TO THIS FUNCTION
    TMF 10/27/23 '''
    
def clean_features(df_in, controls, tracer_df=False):  # a method that drops rows based on conditions
    # Make dataframe copy, create docs in df's image
    df = df_in.copy()
    df['AnySamplesDropped'] = np.nan
    docs = pd.DataFrame().reindex_like(df)
    docs['Mass'] = df['Mass']
    docs['Retention_Time'] = df['Retention_Time']
    docs['Feature_ID'] = df['Feature_ID']
    if tracer_df:
        docs['Tracer_chemical_match'] = df['Tracer_chemical_match']
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
    
    '''REPLICATE FLAG'''
    # Set medians (means in docs) where feature presence is less than some replicate percentage cutoff to nan
    for mean,N in zip(Mean_Samples,Replicate_Percent_Samples):
        docs.loc[((df[N] < controls[0]) & (~df[mean].isnull())), mean] = 'R'
        df.loc[df[N] < controls[0], mean] = np.nan
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
    
    # Create a copy of df prior to CV flag/filter step - this DF will not remove occurrences/features failing CV threshold
    df_flagged = df.copy()
    
    '''CV FLAG'''
    # Create a mask for df based on sample-level CV threshold
    # CV masks
    cv_not_met = pd.DataFrame().reindex_like(df[Mean_Samples])
    for mean,CV in zip(Mean_Samples, CV_Samples):
        #AC Create additional condition such that if CV is no value (i.e. only 1 replicate), the occurrence will not fail (i.e. it passes)
        #cv_not_met[mean] = (df[CV] > controls[1] & ~df[CV].isnull())
        cv_not_met[mean] = df[CV] > controls[1]
    # Create empty cell masks from the docs and df dataframes
    cell_empty = docs[Mean_Samples].isnull()
    cell_empty_df = df[Mean_Samples].isnull()  
    # append CV flag (CV > threshold) to documentation dataframe
    docs[Mean_Samples] = np.where(cv_not_met & cell_empty & ~cell_empty_df, 'CV', docs[Mean_Samples])
    docs[Mean_Samples] = np.where(cv_not_met & ~cell_empty & ~cell_empty_df, docs[Mean_Samples]+', CV', docs[Mean_Samples])
    # Create df[Median_Samples] copy
    m = df[Mean_Samples].copy()
    # Blank out sample medians where occurrence does not meet CV cutoff
    df[Mean_Samples] = m.mask(cv_not_met)  
    
    '''MDL CALCULATION/MASKS'''
    # Calculate feature MDL
    df['BlkStd_cutoff'] = (3 * df[Std_MB[0]]) + df[Mean_MB[0]]
    df['BlkStd_cutoff'] = df['BlkStd_cutoff'].fillna(df[Mean_MB[0]]) # In the case of a single blank replicate, the previous calculation is an empty value as it cannot calculate Std dev; replace with mean value
    df['BlkStd_cutoff'] = df['BlkStd_cutoff'].fillna(0)
    df_flagged['BlkStd_cutoff'] = (3 * df_flagged[Std_MB[0]]) + df_flagged[Mean_MB[0]]
    df_flagged['BlkStd_cutoff'] = df_flagged['BlkStd_cutoff'].fillna(df_flagged[Mean_MB[0]]) # In the case of a single blank replicate, the previous calculation is an empty value as it cannot calculate Std dev; replace with mean value
    df_flagged['BlkStd_cutoff'] = df_flagged['BlkStd_cutoff'].fillna(0)
    docs['BlkStd_cutoff'] = df['BlkStd_cutoff']  
    # Create a mask for docs based on sample-level MDL threshold 
    # Mean Masks
    MDL_sample_mask = pd.DataFrame().reindex_like(df[Mean_Samples])  
    for x in Mean_Samples:
        # Count the number of detects
        MDL_sample_mask[x] = (df[x] > df['BlkStd_cutoff'])

    '''CALCULATE DETECTION COUNTS'''
    # Calculate Detection_Count
    #df['Occurrence_Count(all_samples)'] = MDL_all_mask.sum(axis=1)
    df['Detection_Count(non-blank_samples)'] = MDL_sample_mask.sum(axis=1)
    df_flagged['Detection_Count(non-blank_samples)'] = MDL_sample_mask.sum(axis=1)
    # total number of samples
    mean_samples = len(Mean_Samples)
    # calculate percentage of samples that have a value and store in new column 'Detection_Count(non-blank_samples)(%)'
    df['Detection_Count(non-blank_samples)(%)'] = (df['Detection_Count(non-blank_samples)'] / mean_samples) * 100
    df['Detection_Count(non-blank_samples)(%)'] = df['Detection_Count(non-blank_samples)(%)'].round(1)
    df_flagged['Detection_Count(non-blank_samples)(%)'] = (df_flagged['Detection_Count(non-blank_samples)'] / mean_samples) * 100
    df_flagged['Detection_Count(non-blank_samples)(%)'] = df_flagged['Detection_Count(non-blank_samples)(%)'].round(1)
    # Assign to docs
    docs['Detection_Count(non-blank_samples)'] = df['Detection_Count(non-blank_samples)']
    docs['Detection_Count(non-blank_samples)(%)'] = df['Detection_Count(non-blank_samples)(%)']
        
    '''MDL/ND FLAG'''
    # Update empty cell masks from the docs and df dataframes
    cell_empty = docs[Mean_Samples].isnull()
    cell_empty_df = df[Mean_Samples].isnull() 
    # append ND flag (occurrence < MDL) to documentation dataframe
    docs[Mean_Samples] = np.where(~MDL_sample_mask & cell_empty & ~cell_empty_df, 'ND', docs[Mean_Samples])
    docs[Mean_Samples] = np.where(~MDL_sample_mask & ~cell_empty & ~cell_empty_df, docs[Mean_Samples]+', ND', docs[Mean_Samples])
    
    '''ADD VALUES TO DOC'''
    # Mask, add values back to doc
    values = docs[Mean_Samples].isnull()
    docs[Mean_Samples] = np.where(values, df[Mean_Samples], docs[Mean_Samples])
    
    '''DOCUMENT DROP FEATURES FROM DF'''
    # AC 11/3/2023: Reversing the order of documenting drop features from DF (in cases of overwriting flags, this will match the numbers from the logic tree in theory)
    # Features dropped because no sample is above the detection limit
    docs['Feature_removed'] = np.where((df[Replicate_Percent_MB[0]] != 0) & (df[Mean_Samples].max(axis=1, skipna=True) < df['BlkStd_cutoff']), 'BLK', '')   
    # Features dropped because all samples are below CV threshold
    docs['Feature_removed'] = np.where((df[CV_Samples] > controls[1]).all(axis=1), 'CV', docs['Feature_removed'])
    # Features dropped because all samples are below replicate threshold
    docs['Feature_removed'] = np.where((df[Replicate_Percent_Samples] < controls[0]).all(axis=1), 'R', docs['Feature_removed'])
    # Label features that don't have anything in the blank and have nothing above MRL as removed by CV/R filters
    docs['Feature_removed'] = np.where(((df[Replicate_Percent_MB[0]] == 0) & (df[Mean_Samples].count(axis=1) < 1)), 'CV/R', docs['Feature_removed'])
    
    '''DROP FEATURES FROM DF'''
    # Remove features where all sample abundances are below replicate threshold
    df.drop(df[(df[Replicate_Percent_Samples] < controls[0]).all(axis=1)].index, inplace=True)
    df_flagged.drop(df_flagged[(df_flagged[Replicate_Percent_Samples] < controls[0]).all(axis=1)].index, inplace=True)
    # Remove features where all sample abundances are below CV threshold
    df.drop(df[(df[CV_Samples] > controls[1]).all(axis=1)].index, inplace=True)
    # df_flagged does not drop features due to CV threshold
    
    # Keep samples where the feature doesn't exist in the blank OR at least one sample mean exceeds MDL
    # Remove these features from the feature results
    df = df[((df[Replicate_Percent_MB[0]] == 0) & (df[Mean_Samples].count(axis=1) > 0)) | (df[Mean_Samples].max(axis=1, skipna=True) > df['BlkStd_cutoff'])]
    df_flagged = df_flagged[((df_flagged[Replicate_Percent_MB[0]] == 0) & (df_flagged[Mean_Samples].count(axis=1) > 0)) | (df_flagged[Mean_Samples].max(axis=1, skipna=True) > df_flagged['BlkStd_cutoff'])]
    return df, docs, df_flagged


'''FUNCTIONS FOR COMBINING DATAFRAMES'''

# Combine function for the self.dfs
def combine(df1,df2):
    # Recombine dfs
    if df1 is not None and df2 is not None:
        dfc = pd.concat([df1,df2], sort=True) #fixing pandas FutureWarning
        dfc = dfc.reindex(columns = df1.columns)
    elif df1 is not None:
        dfc = df1.copy()
    else:
        dfc = df2.copy()
    # Get column names
    columns = dfc.columns.values.tolist()
    # Drop duplicates (should not be any)
    dfc = dfc.drop_duplicates(subset=['Mass','Retention_Time'])
    # Get sample Means
    Mean_list =  dfc.columns[(dfc.columns.str.contains(pat ='Mean_')==True)\
                 & (dfc.columns.str.contains(pat ='MB|blank|blanks|BlankSub|_x|_y')==False)].tolist()
    # Count sample-level occurrences and median of means
    dfc['N_Abun_Samples'] = dfc[Mean_list].count(axis=1,numeric_only=True)
    dfc['Mean_Abun_Samples'] = dfc[Mean_list].median(axis=1,skipna=True).round(0)
    # Sort by 'Mass' and 'Retention_Time'
    dfc = dfc[columns].sort_values(['Mass','Retention_Time'],ascending=[True,True])
    return dfc
    

# Combine function for the self.docs and self.dupes
def combine_doc(doc, dupe, tracer_df=False):
    # Get Mean columns
    Mean = doc.columns[doc.columns.str.contains(pat ='Mean_')].tolist()
    # Recombine doc and dupe
    if doc is not None and dupe is not None:
        dupe.loc[:, Mean] = 'D'
        dfc = pd.concat([doc,dupe], sort=True) #fixing pandas FutureWarning
        dfc = dfc.reindex(columns = doc.columns)
    elif doc is not None:
        dfc = doc.copy()
    else:
        dupe.loc[:, Mean] = 'D'
        dfc = dupe.copy()
    # Select columns for keeping, with tracer conditional
    if tracer_df:
        to_keep = ['Feature_ID', 'Mass', 'Retention_Time', 'BlkStd_cutoff', 'AnySamplesDropped', 'Feature_removed', 'Tracer_chemical_match'] + Mean
    else:
        to_keep = ['Feature_ID', 'Mass', 'Retention_Time', 'BlkStd_cutoff', 'AnySamplesDropped', 'Feature_removed'] + Mean
    # Subset with columns to keep; change 'BlkStd_cutoff' to MRL
    dfc = dfc[to_keep]
    dfc.rename({'BlkStd_cutoff':'MRL'}, axis=1, inplace=True)
    
    return dfc
