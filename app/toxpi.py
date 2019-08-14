import numpy as np
import pandas as pd

def process_toxpi(features_df=None, search_df=None, tophit=False, by_mass=True):
    dft = search_df.copy()
    df = features_df.copy()
    TOTAL_ASSAYS = "\/([0-9]+)"  # a regex to find the digits after a slash
    dft['TOTAL_ASSAYS_TESTED'] = dft['TOXCAST_NUMBER_OF_ASSAYS/TOTAL'].str.extract(TOTAL_ASSAYS, expand=True)
    NUMBER_ASSAYS = "([0-9]+)\/"  # a regex to find the digits before a slash
    dft['NUMBER_ACTIVE_ASSAYS'] = dft['TOXCAST_NUMBER_OF_ASSAYS/TOTAL'].str.extract(NUMBER_ASSAYS, expand=True)
    dft = dft.rename(columns={'TOXCAST_PERCENT_ACTIVE': 'PERCENT_ACTIVE_CALLS'})
    dft = dft.rename(columns={'EXPOCAST_MEDIAN_EXPOSURE_PREDICTION_MG/KG-BW/DAY': 'EXPOCAST_MGKG_BWDAY'})
    # dft.drop(['TOXCAST_NUMBER_OF_ASSAYS/TOTAL'],axis=1)
    dft.columns = dft.columns.str.replace(' ', '_')
    dft['DATA_SOURCES_RATIO'] = dft.groupby('INPUT')['DATA_SOURCES'].apply(lambda x: (x / x.max())).round(2)
    df.sort_values('Compound', ascending=True, inplace=True)
    # dft = dft.sort_values('DATA_SOURCES',ascending = False).drop_duplicates('Compound').sort_index()
    df['SEARCHED_MASS'] = df['Mass']
    df['MPP_ASSIGNED_FORMULA'] = df['Compound']
    df['MPP_RETENTION_TIME'] = df['Retention_Time']
    df['FORMULA_MATCH_SCORE'] = df['Score']
    dft['For_Dashboard_Search'] = "1"  # adding this to the join so we only match features that we meant to search
    if by_mass:
        dfe = pd.merge(df, dft, left_on=['SEARCHED_MASS', 'For_Dashboard_Search'],
                       right_on=['INPUT', 'For_Dashboard_Search'], how='left')
        dfe['DASHBOARD_FORMULA_MATCH'] = np.where(dfe['MPP_ASSIGNED_FORMULA'] == dfe['MOLECULAR_FORMULA'], 1, 0)
    else:
        dfe = pd.merge(df, dft, left_on=['Compound', 'For_Dashboard_Search'],
                       right_on=['INPUT', 'For_Dashboard_Search'], how='left')
    if tophit:
        dfe = dfe.drop_duplicates(subset=['Compound', 'Mass', 'Retention_Time', 'Score'])
    else:
        print("Not Selecting Top Hit")
    # print dfe
    columns = df.columns.values.tolist()
    columns.append('INPUT')
    columns.append('FOUND_BY')
    columns.append('DTXSID')
    # dfe.dropna(how='all')
    # dfe = dfe[pd.notnull(dfe['INPUT'])]
    # dfe.fillna('',inplace=True)  # problematic memory use here
    return dfe
