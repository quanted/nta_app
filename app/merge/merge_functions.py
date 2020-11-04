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


# Generate a m/z from the mass found in the NTA results file
###################3 This may be able to be dropped for the WebApp if the MS1 WebApp results have m/z already
def MS1_transform(input_df):
    input_df['m/z'] = np.where(input_df['Ionization_Mode'] == 'Esi+', input_df['Mass']+1.007276, input_df['Mass']-1.007276)
    
    # Rename DTXCID column
    input_df.rename(columns={'DTXCID_INDIVIDUAL_COMPONENT':'DTXCID'}, inplace=True)
    # Extract just the m/z, RT and mode information from the merged stats dataframe for matching purposes
    # Then de-duplicate it
    output_list = list(zip(input_df['m/z'], input_df['Retention_Time'], input_df['Ionization_Mode'], input_df['Mass'], input_df['Feature_ID']))
    output_list = list(set(output_list))    
    return output_list

# Extract just the mass, RT and mode information from the pcdl dataframe for matching purposes
# Then de-duplicate it
def PCDL_transform(input_df):
    # Only keep rows with PCDL matches
    input_df = input_df.dropna(subset=['Score (Lib)'])
    input_df = input_df.reset_index(drop=True)
    
    output_list = list(zip(input_df['Mass'], input_df['RT'], input_df['Ionization_Mode']))
    output_list = list(set(output_list))
    return output_list
    
# Extract just the m/z, RT and mode information from the cfmid dataframe for matching purposes
# Then de-duplicate it    
def CFMID_transform(input_df):
    output_list = list(zip(input_df['MASS_in_MGF'], input_df['RT_in_MGF'], input_df['Ionization_Mode']))
    output_list = list(set(output_list))
    return output_list
    

# Perform mass accuracy/RT window matching between MS1 and CFMID, and MS1 and PCDL    
def Match_results(input_MS1_list, input_cfmid_list, input_pcdl_list, input_RT_accuracy, input_accuracy_ppm):
    
    output_cfmid_match = [] # List to store the matches between MS1 and CFMID MS2
    output_pcdl_match = [] # List to store the matches between MS1 and PCDL MS2   

    # Loop through the MS1 features first before looping through MS2
    for i in range(0, len(MS1_list)):
        #print('MS1 feature', i+1, MS1_list[i])
        # Grab pertinent values from MS1 feature
        MS1_mz = input_MS1_list[i][0]
        MS1_RT = input_MS1_list[i][1]
        MS1_ionization_mode = input_MS1_list[i][2]
        MS1_mass = input_MS1_list[i][3]
        MS1_ID = input_MS1_list[i][4]
        
        # Loop through the MS2 CFMID results for matches to MS1 feature
        for j in range(0, len(input_cfmid_list)):
            
            # Grab pertinent values from MS2 CFMID result
            cfmid_mz = input_cfmid_list[j][0]
            cfmid_RT = input_cfmid_list[j][1]
            cfmid_ionization_mode = input_cfmid_list[j][2]
            
            if MS1_ionization_mode == cfmid_ionization_mode: # Check if the ionization modes match
                if abs(MS1_RT - cfmid_RT) < input_RT_accuracy: # Check if the RT's are close enough
                    if abs((MS1_mz - cfmid_mz)/ MS1_mz *1000000) < input_accuracy_ppm: # Check if the masses are close enough
                        output_cfmid_match.append([MS1_mz, MS1_RT, cfmid_mz, cfmid_RT])
    
        # Loop through the MS2 PCDL results for matches to the MS1 featgure
        for k in range(0, len(input_pcdl_list)):
            
            # Grab pertinent values from MS2 PCDL result
            pcdl_mass = input_pcdl_list[k][0]
            pcdl_RT = input_pcdl_list[k][1]
            pcdl_ionization_mode = input_pcdl_list[k][2]
            
            if MS1_ionization_mode == pcdl_ionization_mode: # Check if the ionization modes match
                if abs(MS1_RT - pcdl_RT) < input_RT_accuracy: # Check if the RT's are close enough
                    if abs((MS1_mass - pcdl_mass)/ MS1_mass * 1000000) < input_accuracy_ppm: # Check if the masses are close enough
                        output_pcdl_match.append([MS1_ID, MS1_RT, pcdl_mass, pcdl_RT])

    return output_cfmid_match, output_pcdl_match


# Concatenate match results back onto MS1 feature dataframe
#def Process_matches(input_cfmid_match, input_pcdl_match, input_cfmid_df, input_pcdl_df, input_MS1):
def Process_matches(input_df, input_RT_accuracy, input_accuracy_ppm):
    
    MS1_list = MS1_transform(input_df[0])
    df_MS1 = input_df[0]
    
    temp_cfmid_neg = input_df[1]
    temp_cfmid_neg['Ionization_Mode'] = 'Esi-'
    temp_cfmid_pos = input_df[2]
    temp_cfmid_pos['Ionization_Mode'] = 'Esi+'
    df_cfmid = pd.concat([temp_cfmid_neg, temp_cfmid_pos])
    cfmid_list = CFMID_transform(df_cfmid)
    
    temp_pcdl_neg = input_df[3]
    temp_pcdl_neg['Ionization_Mode'] = 'Esi-'
    temp_pcdl_pos = input_df[4]
    temp_pcdl_pos['Ionization_Mode'] = 'Esi+'
    df_pcdl = pd.concat([temp_pcdl_neg, temp_pcdl_pos])
    pcdl_list = CFMID_transform(df_cfmid)
    
    
    cfmid_match, pcdl_match = Match_results(MS1_list, cfmid_list, pcdl_list, input_RT_accuracy, input_accuracy_ppm)
    
    # Get back into dataframe space now that the matching is done:
    # 1) Grab the CFMID results (from CFMID dataframe) for the CFMID matches found
    # 2) Grab the MS1 results (from MS1 dataframe) for the CFMID matches found
    df_cfmid_2 = pd.DataFrame(cfmid_match, columns=['m/z', 'MS1_RT', 'MASS_in_MGF', 'RT_in_MGF'])
    df_cfmid_2 = pd.merge(df_cfmid, df_cfmid_2, how='inner', on = ['MASS_in_MGF', 'RT_in_MGF'])
    df_cfmid_2 = pd.merge(df_MS1[['Feature_ID', 'm/z', 'DTXCID', 'DTXSID']], df_cfmid_2[['m/z', 'DTXCID', 'SCORE', 'MASS_in_MGF', 'RT_in_MGF', 'cfmid_file']], how='left', on = ['m/z', 'DTXCID'])
    
    
    # 1) Grab the PCDL results (from PCDL dataframe) for the PCDL matches found
    df_pcdl_2 = pd.DataFrame(pcdl_match, columns=['Feature_ID', 'MS1_RT', 'Mass', 'RT'])
    df_pcdl_2 = pd.merge(df_pcdl_2, df_pcdl, how='left', on = ['Mass', 'RT'])
    df_pcdl_2.rename(columns={'Mass':'pcdl_mass'}, inplace=True) 
    df_pcdl_2.rename(columns={'RT':'pcdl_RT'}, inplace=True)
    df_pcdl_2.rename(columns={'File':'pcdl_file'}, inplace=True)
    df_pcdl_2.rename(columns={'Name':'pcdl_name'}, inplace=True)
    df_pcdl_2.rename(columns={'Score (Lib)':'pcdl_score'}, inplace=True)    


    # Process CFMID results and get the following values:
    # Median CFMID: In a single feature, for a single chemical, there may be multiple CFMID scores
    #   due to having CFMID MS2 results acquired in multiple samples. Get the median of those scores
    # Maximum CFMID median: In a single feature, there are many chemicals, each with median CFMID scores
    #   (see above). Get the maximum median score out of all the chemicals
    # CFMID quotient: Normalize median CFMID scores- divide the median CFMID score by the maximum
    #   CFMID score (see above 2 values). This converts CFMID medians to a normalized value 
    #   that is always between 0 and 1.
    df_cfmid_2['cfmid_score_median'] = df_cfmid_2.groupby(['Feature_ID', 'DTXSID'])['SCORE'].transform('median')
    df_cfmid_2['cfmid_score_median_max'] = df_cfmid_2.groupby('Feature_ID')['cfmid_score_median'].transform('max')
    df_cfmid_2['cfmid_score_median_quot'] = df_cfmid_2['cfmid_score_median']/df_cfmid_2['cfmid_score_median_max']

    # Round values
    df_cfmid_2['SCORE'] = df_cfmid_2['SCORE'].round(3)
    df_cfmid_2['MASS_in_MGF'] = df_cfmid_2['MASS_in_MGF'].round(4)
    df_cfmid_2['RT_in_MGF'] = df_cfmid_2['RT_in_MGF'].round(3)
    
    # Change column types into strings for the subsequent concatenation functions below
    df_cfmid_2['SCORE'] = df_cfmid_2['SCORE'].astype(str)
    df_cfmid_2['MASS_in_MGF'] = df_cfmid_2['MASS_in_MGF'].astype(str)
    df_cfmid_2['RT_in_MGF'] = df_cfmid_2['RT_in_MGF'].astype(str)
    df_cfmid_2['cfmid_file'] = df_cfmid_2['cfmid_file'].astype(str)
    
    # The below performs a concatenation function on when a chemical is observed in multiple MS2 files
    # and thus has multiple matches to a single MS1 feature
    # Rather than having multiple rows for each sample observation, it concatenates sample observations
    # into a single row for each chemical (by grouping by feature and DTXSID, then aggregating the 
    # score, RT, mass and file values into new dataframes)
    df_score = df_cfmid_2.groupby(['Feature_ID', 'DTXSID'], as_index=False)['SCORE'].agg(lambda x: ', '.join(x))
    df_score.rename(columns={'SCORE':'ind_cfmid_score'}, inplace=True)
    df_RT = df_cfmid_2.groupby(['Feature_ID', 'DTXSID'], as_index=False)['RT_in_MGF'].agg(lambda x: ', '.join(x))
    df_RT.rename(columns={'RT_in_MGF':'ind_cfmid_RT'}, inplace=True) 
    df_mass = df_cfmid_2.groupby(['Feature_ID', 'DTXSID'], as_index=False)['MASS_in_MGF'].agg(lambda x: ', '.join(x))
    df_mass.rename(columns={'MASS_in_MGF':'ind_cfmid_mz'}, inplace=True) 
    df_file = df_cfmid_2.groupby(['Feature_ID', 'DTXSID'], as_index=False)['cfmid_file'].agg(lambda x: ', '.join(x))
    df_file.rename(columns={'cfmid_file':'ind_cfmid_file'}, inplace=True) 
    
    # Merge the concatenation dataframes back onto the CFMID dataframe
    df_cfmid_2 = pd.merge(df_cfmid_2, df_score, how='left', on = ['Feature_ID', 'DTXSID'])
    df_cfmid_2 = pd.merge(df_cfmid_2, df_RT, how='left', on = ['Feature_ID', 'DTXSID'])
    df_cfmid_2 = pd.merge(df_cfmid_2, df_mass, how='left', on = ['Feature_ID', 'DTXSID'])
    df_cfmid_2 = pd.merge(df_cfmid_2, df_file, how='left', on = ['Feature_ID', 'DTXSID'])
    
    # Get rid of duplicate rows now that the unique information is preserved in cells separated by commas
    df_cfmid_2 = df_cfmid_2.drop_duplicates(subset=['Feature_ID', 'DTXSID'])
    
    # Process CFMID results and get the following values:
    # CFMID percentile: In a single feature, there are many chemicals with CFMID median values.
    #   Translate those median values into a percentile value for each chemical, which is
    #   normalized to between 0 and 100.
    df_cfmid_2['cfmid_score_median_percentile'] = df_cfmid_2.groupby('Feature_ID')['cfmid_score_median'].rank(pct=True)
    df_cfmid_2['cfmid_score_median_percentile'] = df_cfmid_2['cfmid_score_median_percentile'] * 100

    df_pcdl_2['pcdl_score'] = df_pcdl_2['pcdl_score'].astype(str)
    df_pcdl_2['pcdl_mass'] = df_pcdl_2['pcdl_mass'].astype(str)
    df_pcdl_2['pcdl_RT'] = df_pcdl_2['pcdl_RT'].astype(str)
    
    df_mass = df_pcdl_2.groupby('Feature_ID')['pcdl_mass'].agg(lambda x: ', '.join(x))
    df_RT = df_pcdl_2.groupby('Feature_ID')['pcdl_RT'].agg(lambda x: ', '.join(x))
    df_name = df_pcdl_2.groupby('Feature_ID')['pcdl_name'].agg(lambda x: ', '.join(x))
    df_file = df_pcdl_2.groupby('Feature_ID')['pcdl_file'].agg(lambda x: ', '.join(x))
    df_score = df_pcdl_2.groupby('Feature_ID')['pcdl_score'].agg(lambda x: ', '.join(x))
    
    # Bring the concatenated series together into a dataframe for merging
    df_pcdl_2_concat = pd.DataFrame({'Feature_ID':df_mass.index, 'ind_pcdl_mass':df_mass.values, 
                                  'ind_pcdl_RT':df_RT.values, 'ind_pcdl_name':df_name.values,
                                  'ind_pcdl_file':df_file.values, 'ind_pcdl_score':df_score.values})

    df_output = pd.merge(df_MS1, df_pcdl_2_concat, how='left', on='Feature_ID')
        
    df_output = pd.merge(df_output, df_cfmid_2[['Feature_ID', 'DTXSID', 'cfmid_score_median','cfmid_score_median_max',
                                          'cfmid_score_median_quot','cfmid_score_median_percentile',
                                          'ind_cfmid_score','ind_cfmid_RT','ind_cfmid_mz','ind_cfmid_file']],
                                            how='left', on=['Feature_ID', 'DTXSID'])

    return df_output