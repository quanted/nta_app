#!/usr/bin/env python

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as pt
from matplotlib.pyplot import figure, show, rc
import pandas as pd

WA=1.0
WE=1.0
WN=2.0
WB=2.0

def plot_toxpi(Radii=[]):
# force square figure and square axes looks better for polar, IMO
    fig = pt.figure(facecolor='white')
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(False)
    #fig.patch.set_visible(False)

    Theta=[0,2*np.pi/3,4*np.pi/3,5*np.pi/3]
    Width=[2*np.pi/3,2*np.pi/3,np.pi/3,np.pi/3]
    facecolor=['dodgerblue','m','limegreen','orange']
    annotation=['Detection \n Frequency','Bioactivity','Exposure','Abundance']
    text_pos=[np.pi/4,np.pi,2.92*np.pi/2,7.3*np.pi/4]
    HA=['left','right','left','left']
    #VA=['top','top','bottom','bottom']
    for i in range(0,4):
        theta = Theta[i]
        bars = ax.bar(Theta[i], Radii[i], width=Width[i], bottom=0.0,edgecolor="none")
        ax.annotate(annotation[i],xy=(Theta[i],max(Radii)),xytext=(text_pos[i],5.6),horizontalalignment=HA[i],size=14)
        for bar in bars:
             bar.set_facecolor(facecolor[i])
    pt.show()


def classification(x=None,y=None):
    ratio = x/x.max()
    if ratio ==1.0 and y!=None:
        clas = 'A1'
    if ratio <1.0 and y!=None:
        clas = 'A2'
    if ratio ==1.0 and y==None:
        clas = 'B1'
    if ratio <1.0 and y==None:
        clas = 'B2'
    return clas    


def process_toxpi(df=None, dir='', file='',tophit=False,by_mass=True):
    #xls_file = pd.ExcelFile(file)
    #dft = xls_file.parse('Worksheet1', index_col=None, na_values='-')
    dft = pd.read_csv(dir+"/"+file,sep='\t',na_values= '-')
    #df = pd.read_csv('L:\Lab\NERL_RTP_D589A_Quincy\Hussein\Python_NTA_v2\\trial_3_hussein\\499_POS_Combined.csv')
        #dft = pd.read_excel(file,'Worksheet1',index_col=None)
    directory = dir
    # Some initial file cleaning
    TOTAL_ASSAYS = "\/([0-9]+)" # a regex to find the digits after a slash 
    dft['TOTAL_ASSAYS_TESTED'] = dft['TOXCAST_NUMBER_OF_ASSAYS/TOTAL'].str.extract(TOTAL_ASSAYS,expand=True)
    NUMBER_ASSAYS = "([0-9]+)\/" # a regex to find the digits before a slash
    dft['NUMBER_ACTIVE_ASSAYS'] = dft['TOXCAST_NUMBER_OF_ASSAYS/TOTAL'].str.extract(NUMBER_ASSAYS,expand=True)
    dft = dft.rename(columns = {'TOXCAST_PERCENT_ACTIVE':'PERCENT_ACTIVE_CALLS'})
    dft = dft.rename(columns = {'EXPOCAST_MEDIAN_EXPOSURE_PREDICTION_MG/KG-BW/DAY':'EXPOCAST_MGKG_BWDAY'})
    #dft.drop(['TOXCAST_NUMBER_OF_ASSAYS/TOTAL'],axis=1)
    dft.columns = dft.columns.str.replace(' ','_')
    dft['DATA_SOURCES_RATIO'] = dft.groupby('INPUT')['DATA_SOURCES'].apply(lambda x: (x/x.max())).round(2)
    if by_mass:
        dft['INPUT'] = dft['INPUT'].str.replace("\ \+\/\- .*$","")
        dft['INPUT'] = dft['INPUT'].astype(float)

        
    
    
    '''
    #Set a letter+number category for each compound
    dft['NUMBER_CATEGORY']='A'
    PAC_none=dft['PERCENT_ACTIVE_CALLS'] != dft['PERCENT_ACTIVE_CALLS'] #Pythonic way to find if value is None
    EMB_none=dft['EXPOCAST_MGKG_BWDAY'] != dft['EXPOCAST_MGKG_BWDAY']
    dft.loc[(dft['DATA_SOURCES_RATIO']==1) & (PAC_none==False) & (EMB_none==False),'NUMBER_CATEGORY']='A1'
    dft.loc[(dft['DATA_SOURCES_RATIO']<1) & (PAC_none==False) & (EMB_none==False),'NUMBER_CATEGORY']='A2'
    dft.loc[(dft['DATA_SOURCES_RATIO']==1) & (PAC_none) & (EMB_none),'NUMBER_CATEGORY']='B1'
    dft.loc[(dft['DATA_SOURCES_RATIO']<1) & (PAC_none) & (EMB_none),'NUMBER_CATEGORY']='B2'

    #Exposure Catergorization
    dft['EXPOSURE_CATEGORY']=1
    dft.loc[(dft['EXPOCAST_MGKG_BWDAY'] != dft['EXPOCAST_MGKG_BWDAY']),'EXPOSURE_CATEGORY']=None
    dft.loc[(dft['EXPOCAST_MGKG_BWDAY'] == 0) & (dft['EXPOCAST_MGKG_BWDAY'] < 10**(-8)),'EXPOSURE_CATEGORY']=1  

    for i in range(2,8):
        dft.loc[(dft['EXPOCAST_MGKG_BWDAY'] >= 10**(i-10)) & (dft['EXPOCAST_MGKG_BWDAY'] < 10**(i-9)),'EXPOSURE_CATEGORY']=i
    '''
    df.sort_values('Compound',ascending = True, inplace=True)
    #dft = dft.sort_values('DATA_SOURCES',ascending = False).drop_duplicates('Compound').sort_index()
    df['SEARCHED_MASS'] = df['Mass']
    df['MPP_ASSIGNED_FORMULA'] = df['Compound']
    df['MPP_RETENTION_TIME'] = df['Retention_Time']
    df['FORMULA_MATCH_SCORE'] = df['Score']
    if by_mass:
        dfe = pd.merge(df,dft,left_on='SEARCHED_MASS',right_on='INPUT',how='left')
        dfe['DASHBOARD_FORMULA_MATCH'] = np.where(dfe['MPP_ASSIGNED_FORMULA'] == dfe['MOLECULAR_FORMULA'],1,0)
    else:
        dfe = pd.merge(df,dft,left_on='Compound',right_on='INPUT',how='left')          
    if tophit:
        dfe = dfe.drop_duplicates(subset=['Compound','Mass','Retention_Time','Score'])
    else:
        print("Not Selecting Top Hit")
    #print dfe
    columns = df.columns.values.tolist()
    columns.append('INPUT')    
    columns.append('FOUND_BY')
    columns.append('DTXSID')
    print(columns)
    dfe.dropna(how='all')
    dfe = dfe[pd.notnull(dfe['INPUT'])]  
    #dfe.fillna('',inplace=True)
    #dfe = dfe.set_index(columns)
    #dfe.to_csv(directory+"/csv_toxpi.csv")

    #print dfe
    #dfe.to_excel('trying_excel.xlsx',na_rep='',engine='xlsxwriter') 
    return dfe


def calculate_toxpi(df,dir):
    directory = dir
    #df = pd.read_csv('/home/hussein/Documents/NTA/Python_alt/Trial9_Dawn/MPP_output_Dust_Mapping_2_Neg_toxpi.csv')
    df = df[pd.notnull(df['CASRN'])]
    df['TP_BIOACTIVITY'] = (df['PERCENT_ACTIVE_CALLS'] - df['PERCENT_ACTIVE_CALLS'].min())/\
                (df['PERCENT_ACTIVE_CALLS'].max() - df['PERCENT_ACTIVE_CALLS'].min())

    df['TP_EXPOSURE'] = (df['EXPOSURE_CATEGORY'] - df['EXPOSURE_CATEGORY'].min())/\
                (df['EXPOSURE_CATEGORY'].max() - df['EXPOSURE_CATEGORY'].min())    

    df['TP_FREQUENCY'] = (df['N_Abun_Samples'] - df['N_Abun_Samples'].min())/\
                (df['N_Abun_Samples'].max() - df['N_Abun_Samples'].min())

    df['TP_ABUNDANCE'] = (np.log(df['Median_Abun_Samples']) - np.log(df['Median_Abun_Samples']).min())/\
                (np.log(df['Median_Abun_Samples']).max() - np.log(df['Median_Abun_Samples']).min())

    df['TOXPI_SCORE'] = (WB*df['TP_BIOACTIVITY']+WE*df['TP_EXPOSURE']+WA*df['TP_ABUNDANCE']+WN*df['TP_FREQUENCY']).where(df['NUMBER_CATEGORY']=='A1',None)

    print(df)
    df.to_csv(directory+"/calculated_toxpi_variables.csv", index=False)
    return df




#dire='L:/Lab/NERL_RTP_D589A_Quincy/Hussein/Python_NTA_v2\/trial_3_hussein'
#filee='ChemistryDashboard-Batch-Search_2018-05-24_16_38_36.tsv'
#fix='/home/hussein/Documents/NTA/Python_alt/Trial2_Hussein/ChemistryDashboard-AdvancedSearch_2017-12-07_14_46_46.xls'
#fi='/home/hussein/Documents/NTA/Python_alt/ChemistryDashboard-AdvancedSearch_2017-12-07_14_46_46.csv'    


#process_toxpi(dir=dire,file=filee)
#calculate_toxpi(None,dir)
#Radii=[2,1,3,5]

#plot_toxpi(Radii)
