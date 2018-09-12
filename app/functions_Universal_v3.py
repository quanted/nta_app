# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:11:13 2017

@author: HALGhoul
"""

import pandas as pd
import numpy as np
import re
import os
from operator import itemgetter
from itertools import groupby
from difflib import SequenceMatcher

#REP_NUM = 3
HBR = 3.0 # High_Blank_Ratio condition
HMR = 1.5 # High_Mid_Ratio condition
SCORE = 90 # formula match is 90


def fix_names(df,index): # parse the Dataframe into a numpy array
        #df.columns = df.columns.str.replace(': Log2','') #log specific code
        df.columns = df.columns.str.replace(' ','_')
        df.columns = df.columns.str.replace('\([^)]*\)','')
        df['Compound'] = df['Compound'].str.replace("\ Esi.*$","")
        if 'Ionization_mode' in df.columns:
            df.rename(columns = {'Ionization_mode':'Ionization_Mode'},inplace=True)
        #df.drop(['CompositeSpectrum','Compound_Name'],axis=1)
        df.drop(['Compound_Name'],axis=1)
        Headers = parse_headers(df,index)
        Abundance = [item for sublist in Headers for item in sublist if len(sublist)>1]    
        Samples= [x for x in Abundance]
        NewSamples = common_substrings(Samples)
        df.drop([col for col in df.columns if 'Spectrum' in col], axis=1,inplace=True)
        for i in range(len(Samples)):
            df.rename(columns = {Samples[i]:NewSamples[i]},inplace=True)    
        #df = df
        return df





def read_data(file,index):  # read a csv file into a DataFrame
        ext = os.path.splitext(file)[1]
        print(ext)
        if ext == '.tsv':
            df = pd.read_csv(file,sep='\t',comment='#',na_values= 1 | 0)
        if ext == '.csv':
            df = pd.read_csv(file,comment='#',na_values= 1 | 0)
        df = fix_names(df,index)
        return df





def differences(s1,s2): #find the number of different characters between two strings (headers)
        s1 = re.sub(re.compile(r'\([^)]*\)'),'',s1)
        s2 = re.sub(re.compile(r'\([^)]*\)'),'',s2)    
        count = sum(1 for a, b in zip(s1, s2) if a != b) + abs(len(s1) - len(s2))
        return count




def formulas(df):
        df.drop_duplicates(subset='Compound',keep='first',inplace=True)
        formulas = df.loc[df['For_Dashboard_Search'] == '1','Compound'].values #only features flagged for Dashboard search
        print(formulas)
        return formulas


def masses(df):
        #df.drop_duplicates(subset='Mass',keep='first',inplace=True)
        masses = df.loc[df['For_Dashboard_Search'] == '1','Mass'].values #only features flagged for Dashboard search
        print(masses)
        return masses


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
        print((Headers[1]))
        return Headers[index]





def score(df): # Get that Sneaky score from Annotations.
        regex = "^.*=(.*) \].*$" # a regex to find the score looking for a pattern of "=something_to_find ]" 
        if "Annotations" in df:
            if df.Annotations.str.contains('overall=').any():
                df['Score'] = df.Annotations.str.extract(regex,expand=True).astype('float64')
        else:
            df['Score'] = None
        return df





def statistics(df,index): # calculate Mean,Median,STD,CV for every feature in a sample of multiple replicates
        Abundance = [[],[]]
        Headers = [0,0]
        headers = [0,0]
        Headers[index] = parse_headers(df,index)
        Abundance[index] = [item for sublist in Headers[index] for item in sublist if len(sublist)>1]
        df = score(df) 
    # Do some statistical acrobatics
        headers[index] = ['Compound','Ionization_Mode','Score','Mass','Retention_Time','Frequency'] + Abundance[index]
        df = df[headers[index]]
        print((Headers[index])) #stopped here before my optometrist appointment
        for list in Headers[index]:
                REP_NUM = len(list)
                if REP_NUM > 1:
                    for i in range(0,REP_NUM):
                        # the match part finds the indices of the largest common subtring between two strings
                            match = SequenceMatcher(None, list[i], list[i+1]).find_longest_match(0, len(list[i]),0, len(list[i+1]))
                            df['Mean_'+ str(list[i])[match.a:match.a +  match.size]] = df[list[i:i + REP_NUM]].mean(axis=1).round(0)
                            df['Median_'+ str(list[i])[match.a:match.a +  match.size]] = df[list[i:i + REP_NUM]].median(axis=1,skipna=True).round(0) 
                            df['STD_'+ str(list[i])[match.a:match.a +  match.size]] = df[list[i:i + REP_NUM]].std(axis=1,skipna=True).round(0)
                            df['CV_'+ str(list[i])[match.a:match.a +  match.size]] = (df['STD_'+ str(list[i])[match.a:match.a +  match.size]]/df['Mean_'+ str(list[i])[match.a:match.a +  match.size]]).round(2)            
                            df['N_Abun_'+ str(list[i])[match.a:match.a +  match.size]] = df[list[i:i + REP_NUM]].count(axis=1).round(0)
                            #print list[i][match.a:match.a +  match.size]
                            break
        df.sort_values(['Mass','Retention_Time'],ascending=[True,True],inplace=True)    
        #df.to_csv('input-updated.csv', index=False)
        return df



def Blank_Subtract(df,index):
        Abundance = [[],[]]
        Headers = [0,0]
        Blanks = [[],[]]
        Median = [[],[]]
        Headers[index] = parse_headers(df,index)
        Abundance[index] = [item for sublist in Headers[index] for item in sublist if len(sublist)>1]

        # On with the agony of subtracting the MB median from Samples
        Blanks[index] = df.columns[df.columns.str.contains(pat ='MB_|blank|blanks|BLANK|Blank')].tolist()
        Median[index] =  df.columns[(df.columns.str.contains(pat ='Median_')==True) & (df.columns.str.contains(pat ='MB|blank|blanks|BLANK|Blank')==False)].tolist()
        df['Median_ALLMB'] = df[Blanks[index]].median(axis=1,skipna=True).round(0).fillna(0)
        df[Abundance[index]] = df[Abundance[index]].sub(df['Median_ALLMB'],axis=0) #subtract the median of MBs from every Sample median
        for median in Median[index]:
            df["BlankSub_"+str(median)] = df[median].sub(df['Median_ALLMB'],axis=0)
            df["BlankSub_"+str(median)] = df["BlankSub_"+str(median)].clip(lower=0).replace({0:np.nan})
        df[Abundance[index]] = df[Abundance[index]].clip(lower=0).replace({0:np.nan})
        return df


 
def check_feature_tracers(df,tracers_file,Mass_Difference,Retention_Difference,ppm): #a method to query and save the features with tracers criteria
        df1 = df
        df2 = pd.read_csv(tracers_file,comment='#',na_values= 1 | 0)
        #b_Statistics[index] = [B + '_x' for B in Statistics[index]]
        df2['Rounded_Mass'] = df2['Monoisotopic_Mass'].round(0)
        #df2['Rounded_RT'] = df2['Retention_Time'].round(0)
        df1.rename(columns = {'Mass':'Observed_Mass','Retention_Time':'Observed_Retention_Time'},inplace=True)
        df1['Rounded_Mass'] = df1['Observed_Mass'].round(0)
        #df['Rounded_RT'] = df['Observed_Retention_Time'].round(0)
        dft = pd.merge(df2,df1,how='left',on=['Rounded_Mass','Ionization_Mode'])
        if ppm:
            dft['Matches'] = np.where((abs((dft['Monoisotopic_Mass']-dft['Observed_Mass'])/dft['Monoisotopic_Mass'])*1000000<=Mass_Difference) & (abs(dft['Retention_Time']-dft['Observed_Retention_Time'])<=Retention_Difference) ,1,0)
        else:    
            dft['Matches'] = np.where((abs(dft['Monoisotopic_Mass']-dft['Observed_Mass'])<=Mass_Difference) & (abs(dft['Retention_Time']-dft['Observed_Retention_Time'])<=Retention_Difference) ,1,0)
        dft = dft[dft['Matches']==1]    
        dft.drop(['Rounded_Mass','Matches'],axis=1,inplace=True)
        df.rename(columns = {'Observed_Mass':'Mass','Observed_Retention_Time':'Retention_Time'},inplace=True)        
        return dft
        
        '''
        df_sql = [None,None]
        if ppm: # PPM cut
            print "PPM selected"
            mass_cut =  """ from df2 as a left join df1 as b where abs((a.Monoisotopic_Mass-b.Mass)/a.Monoisotopic_Mass)*(1000000)<=""" + str(Mass_Difference)
        else: # Da Cut
            print "Da selected"            
            mass_cut =  """ from df2 as a left join df1 as b where abs(a.Monoisotopic_Mass-b.Mass)<=""" + str(Mass_Difference)            
        Statistics = [[],[]]
        b_Statistics = [[],[]]
        q = [None,None]
        df1 = df    
        df2 = pd.read_csv(os.getcwd()+"\Tracers_Table_ for_SRM2585_20170524.csv",comment='#',na_values= 1 | 0)
        Statistics[index] = df.columns[df.columns.str.contains(pat ='N_|CV_|Mean_|Median_|STD_')].tolist()
        print df2
        b_Statistics[index] = ["b." + B for B in Statistics[index]]
        q[index] ="""select a.*, b.Mass as Observed_Mass,
                 b.Retention_Time as Observed_Retention_Time,""" + " , ".join([b + " as " + a  for b,a in zip(b_Statistics[index],Statistics[index])]) + mass_cut  + """ and abs(a.Retention_Time-b.Retention_Time)<=""" + str(Retention_Difference) + """ and a.Ionization_Mode = b.Ionization_Mode;"""
        df_sql[index] = sqldf(q[index],locals())         
        #df_sql.to_csv('input_after_tracers.csv', index=False)
        return df_sql[index]
        '''




#def clean_features(df,index,ENTACT): # a method that drops rows based on conditions
def clean_features(df,index,ENTACT,controls): # a method that drops rows based on conditions
    
        Abundance=[[],[]]
        Abundance[index] =  df.columns[df.columns.str.contains(pat ='N_Abun_')].tolist()   
        #for header in Abundance:
        #    df = df.drop(df[df[header] < 2].index) # drop rows with n_abundance_high <2
  
        Median=[[],[]]
        Median_Samples=[[],[]]       
        Median_MB=[[],[]]
        Median_Low = [[],[]]
        Median_Mid = [[],[]]
        Median_High=[[],[]]
        N_Abun_High = [[],[]]
        N_Abun_MB = [[],[]]
        N_Abun_Samples = [[],[]]
        CV = [[],[]]
        CV_Samples = [[],[]]
        blanks = ['MB','mb','mB','Mb','blank','Blank','BLANK']        
        Median[index] =  df.columns[df.columns.str.contains(pat ='Median_')].tolist()
        Median_Samples[index] =  [md for md in Median[index] if not any(x in md for x in blanks)]   
       #ENTACT specific code
        Median_High[index] = [md for md in Median[index] if 'C' in md] 
        Median_Mid[index] = [md for md in Median[index] if 'B' in md]
        Median_Low[index] = [md for md in Median[index] if 'A' in md]
        Median_MB[index] = [md for md in Median[index] if any(x in md for x in blanks)]
        print("***********")
        print((Median_MB[index]))
        N_Abun_High[index] = [N for N in Abundance[index] if 'C' in N]
        N_Abun_MB[index] = [N for N in Abundance[index] if 'MB' in N]
        N_Abun_Samples[index] = [N for N in Abundance[index] if not any(x in N for x in blanks)]    
        N_Abun_MB[index] = [N for N in Abundance[index] if 'MB' in N]
        CV[index] =  df.columns[df.columns.str.contains(pat ='CV_')].tolist()        
        CV_Samples[index] = [C for C in CV[index] if not any(x in C for x in blanks)]        
        print("***********")
        print((N_Abun_Samples[index]))   
        
        if ENTACT: # ENTACT data cleaning
            for median in Median_High[index]:
                df['HightoMid_ratio']=df[median].astype('float')/df[Median_Mid[index][0]].astype('float')
                df['HightoBlanks_ratio']=df[median].astype('float')/df[Median_MB[index][0]].astype('float')
                df.drop(df[df[N_Abun_High[index][0]] < controls[1]].index,inplace=True)
                df = df[(df['HightoBlanks_ratio'] >= controls[0]) | (df[N_Abun_MB[index][0]] == 0)]  
                df = df[(df[median].astype('float')/df[Median_Mid[index][0]].astype('float') >= controls[2]) | 
                           (df['HightoMid_ratio'].isnull())]
                
        else: # Regular NTA data
            #set medians where feature abundance is less than some cutoff to nan
            for median,N in zip(Median_Samples[index],N_Abun_Samples[index]):
                print((str(median) + " , " +str(N)))       
                df.loc[df[N]<controls[1],median]= np.nan
          #find the median of all samples and select features where median_samples/ median_blanks >= cutoff
            df['Median_ALLSamples'] = df[Median_Samples[index]].median(axis=1,skipna=True).round(0)
            df['SampletoBlanks_ratio']=df['Median_ALLSamples'].astype('float')/df[Median_MB[index][0]].astype('float')
            df = df[(df[N_Abun_MB[index][0]] == 0) | (df['SampletoBlanks_ratio'] >= controls[0])] 
          # remove all features where the abundance is less than some cutoff in all samples  
            df.drop(df[(df[N_Abun_Samples[index]] < controls[1]).all(axis=1)].index,inplace=True)
            df.drop(df[(df[CV_Samples[index]] >= controls[2]).all(axis=1)].index,inplace=True)
            
        return df
  



    
def flags(df): # a method to develop required flags
        df['Neg_Mass_Defect'] = np.where((df.Mass - df.Mass.round(0)) < 0 , '1','0')
        df['Halogen'] = np.where(df.Compound.str.contains('F|l|r|I'),'1','0')
        df['Formula_Match'] = np.where(df.Score != df.Score,'0','1') #check if it does not have a score
        df['Formula_Match_Above90'] = np.where(df.Score >= SCORE,'1','0')
        df['X_NegMassDef_Below90'] = np.where(((df.Score < SCORE) & (df.Neg_Mass_Defect == '1') & (df.Halogen == '1')),'1','0')
        df['For_Dashboard_Search'] = np.where(((df.Formula_Match_Above90 == '1') | (df.X_NegMassDef_Below90 == '1')) , '1', '0') 
        df.sort_values(['Formula_Match','For_Dashboard_Search','Formula_Match_Above90','X_NegMassDef_Below90'],ascending=[False,False,False,False],inplace=True) 
        #df.to_csv('input-afterflag.csv', index=False) 
        #print df1 
        df.sort_values('Compound',ascending=True,inplace=True)
        return df



def match_headers(list1=None,list2=None):
        string_match = list()
        print((len(list1), len(list2)))
        for i in range(len(list2)):
            print((list1[i] + "  ,  " + list2[i])) 
            string_match.append("".join([list2[i][j] for j, (a,b) in enumerate(zip(list1[i],list2[i])) if a == b]))    
        print((len(string_match)))
        return string_match    

def append_headers(list1,list2):
        #list1.sort()
        list_new = list()
        #list2.sort()
        diff = list()
        if len(list1) > len(list2):
            diff = list(set(list1) - set(list2))
            list_new = list2 + diff        
        if len(list2) > len(list1):
            diff = list(set(list2) - set(list1))
            list_new = list1 + diff
        else:
            list_new = list1
        print(list_new)
        return diff            
        

def common_substrings(ls=[]):
        match  = SequenceMatcher(None,ls[0],ls[len(ls)-1]).find_longest_match(0,len(ls[0]),0,len(ls[len(ls)-1]))
        common = ls[0][match.a: match.a + match.size]
        print((" ********* " + common))
        lsnew = list()
        for i in range(len(ls)):
            if len(common) > 3:
                lsnew.append(ls[i].replace(common,''))
            else:
                lsnew.append(ls[i])            
        #print ls
        return lsnew
    


def combine(df1,df2):
    #Headers = [[],[]]
        #Headers[0] = parse_headers(df1,0)
        #Headers[1] = parse_headers(df2,1)
    print("##############")
    Abundance=[[],[]]
    Abundance[0] = df1.columns.values.tolist()
    Abundance[1] = df2.columns.values.tolist()         
    diff = append_headers(Abundance[0],Abundance[1])
    #print len(df1.columns.values.tolist())
    #for i in range(len(Abundance[0])):
    #    #print (Abundance[0][i],Abundance[1][i])
    #    df1.rename(columns = {Abundance[0][i]:new_headers[i]},inplace=True)
    #    df2.rename(columns = {Abundance[1][i]:new_headers[i]},inplace=True)
    #print df1.columns.values.tolist()
    print(" ||||___|||| - - - - - - ")
    #print df2.columns.values.tolist()
    #df1[list(set(Abundance[0])-set(diff))]    = np.nan
    #df2[list(set(Abundance[1])-set(diff))]    = np.nan
    dfc = pd.concat([df1,df2])
    dfc = dfc.reindex_axis(df1.columns, axis=1)
    columns = dfc.columns.values.tolist()
    print((str(len(columns)) + " ##### " + str(len(df1.columns.values.tolist())) + " #### " + str(len(df2.columns.values.tolist()))))
    dfc = pd.merge(dfc,df2,suffixes=['','_x'],on='Compound',how='left')
    dfc = pd.merge(dfc,df1,suffixes=['','_y'],on='Compound',how='left')

    # create new flags
    dfc = dfc.drop_duplicates(subset=['Compound','Mass','Retention_Time','Score'])
    dfc['Both_Modes'] = np.where(((abs(dfc.Mass_x-dfc.Mass_y)<=0.005) & (abs(dfc.Retention_Time_x-dfc.Retention_Time_y)<=1)),'1','0')
    dfc['N_Compound_Hits'] = dfc.groupby('Compound')['Compound'].transform('size')
    Median_list =  dfc.columns[(dfc.columns.str.contains(pat ='Median_')==True)\
                 & (dfc.columns.str.contains(pat ='MB|blank|blanks|BlankSub|_x|_y')==False)].tolist()
    print(Median_list)     
    dfc['N_Abun_Samples'] = dfc[Median_list].count(axis=1,numeric_only=True)
    dfc['Median_Abun_Samples'] = dfc[Median_list].median(axis=1,skipna=True).round(0)
    dfc['One_Mode_No_Isomers'] = np.where(((dfc.Both_Modes == '0') & (dfc.N_Compound_Hits == 1)),'1','0')
    dfc['One_Mode_Isomers'] = np.where(((dfc.Both_Modes == '0') & (dfc.N_Compound_Hits > 1)),'1','0')
    dfc['Two_Modes_No_Isomers'] = np.where(((dfc.Both_Modes == '1') & (dfc.N_Compound_Hits == 2)),'1','0')
    dfc['Two_Modes_Isomers'] = np.where(((dfc.Both_Modes == '1') & (dfc.N_Compound_Hits > 2)),'1','0')
    dfc['Est_Chem_Count'] = None #Default to non-type
    dfc.loc[dfc['One_Mode_No_Isomers'] == '1','Est_Chem_Count'] = 1
    dfc.loc[dfc['One_Mode_Isomers'] == '1','Est_Chem_Count'] = dfc['N_Compound_Hits']
    dfc.loc[(dfc['Two_Modes_No_Isomers'] == '1') | (dfc['Two_Modes_Isomers'] == '1'),'Est_Chem_Count'] = dfc['N_Compound_Hits']/2    
    columns.extend(('Both_Modes','N_Compound_Hits','N_Abun_Samples','Median_Abun_Samples','One_Mode_No_Isomers','One_Mode_Isomers','Two_Modes_No_Isomers',
            'Two_Modes_Isomers','Est_Chem_Count'))
    dfc = dfc[columns].sort_values(['Compound'],ascending=[True])

    #dft.reset_index() 
    #dft.dropna(inplace=True)
    return dfc
    



def reduce(df,index):
        Abundance = [[],[]]
        Headers = [0,0]
        Headers[index] = parse_headers(df,index)
        Abundance[index] = [item for sublist in Headers[index] for item in sublist if len(sublist)>2]         
        df.drop(Abundance[index],axis=1,inplace=True)
        return df


def adduct_identifier(df,index,Mass_Difference,Retention_Difference,ppm):
    columns = df.columns.values.tolist()
    print(("type is " + str(type(Mass_Difference))))

    df['rt_rounded'] = df['Retention_Time'].round(2)
    dft = pd.merge(df,df,how='left',suffixes = ('','_y'), on='rt_rounded')
    d = {'Formate':['Esi-',43.99093],'Na':['Esi+',21.98194],'Ammonium':['Esi+',17.02655],'H2O':['Esi-',17.00329],'CO2':['Esi-',42.98255]} #dictionary of adducts
    lst = list()
    boolst = list()
    for key in d:    
        is_name = 'is_' + str(key) + '_Adduct'
        has_name = 'has_' + str(key) + '_Adduct'
        print((d[key][1]))
        if ppm: # PPM cut
            print("PPM selected")            
            dft[is_name] = np.where( (abs(dft.Retention_Time-dft.Retention_Time_y)<Retention_Difference) & (dft.Ionization_Mode==d[key][0])\
                     & (((abs(dft.Mass-(dft.Mass_y+d[key][1]))/dft.Mass)*10**6)<=Mass_Difference),'1','')
        else: # Da Cut
            print("Da selected")            
            dft[is_name] = np.where( (abs(dft.Retention_Time-dft.Retention_Time_y)<Retention_Difference) & (dft.Ionization_Mode==d[key][0])\
                     & ((abs(dft.Mass-(dft.Mass_y+d[key][1])))<=Mass_Difference),'1','')

        dft[has_name] = np.where( (abs(dft.Retention_Time-dft.Retention_Time_y)<Retention_Difference) & (dft.Ionization_Mode==d[key][0])\
                 & (((abs(dft.Mass-(dft.Mass_y-d[key][1]))/dft.Mass)*10**6)<=Mass_Difference),'1','')
        dft['temp_'+str(key)+'_category'] = None        
        dft['temp_'+str(key)+'_RTdiff']= None        
        dft['temp_'+str(key)+'_Massdiff']= None                
        dft.loc[(dft[is_name] =='1') | (dft[has_name] =='1'),'temp_'+str(key)+'_category' ] = 1       
        dft.loc[((dft[is_name] =='1') & (dft[is_name].notnull())) | ((dft[has_name] =='1') & (dft[has_name].notnull())),'temp_'+str(key)+'_RTdiff'] = abs(dft.Retention_Time-dft.Retention_Time_y)
        dft.loc[((dft[is_name] =='1') & (dft[is_name].notnull())) | ((dft[has_name] =='1') & (dft[has_name].notnull())),'temp_'+str(key)+'_Massdiff'] = abs(dft.Mass-dft.Mass_y)
        dft['unique_'+str(key)+'_Number'] = dft.groupby(['temp_'+str(key)+'_category','temp_'+str(key)+'_Massdiff','temp_'+str(key)+'_RTdiff']).ngroup()
        dft.loc[dft['unique_'+str(key)+'_Number'] < 0,'unique_'+str(key)+'_Number' ] = np.nan
        lst.extend((is_name,has_name,'unique_'+str(key)+'_Number'))
        boolst.extend((True,True,True))

    dft.sort_values(lst,ascending=boolst,inplace=True)
    #if index==0:
    #    dft.to_csv('adduct_trial.csv')    
    dft.drop_duplicates(subset=['Compound','Mass','Retention_Time'],keep='last',inplace=True)
    columns.extend(lst)
    dft = dft[columns]
    
    print(dft)
    return dft    


def duplicates(df,index):
        Abundance = [[],[]]
        a_Abundance = [[],[]]
        b_Abundance = [[],[]]
        Abundance[index] = [item for sublist in parse_headers(df,index) for item in sublist if len(sublist)>1]
        #a_Abundance[index] = [ A for A in Abundance[index] if 'MB' not in A]
       #b_Abundance[index] = [ A + "_x" for A in Abundance[index] if 'MB' not in A]
        a_Abundance[index] = [ A for A in Abundance[index]]
        b_Abundance[index] = [ A + "_x" for A in Abundance[index]]
        print((a_Abundance[1]))

        df['Feature_Number'] = df.index
        df['Mass_Rounded'] = df['Mass'].round(2)
        df['Retention_Time_Rounded'] = df['Retention_Time'].round(1)
        df2 = pd.merge(df,df,how='left',suffixes=('','_x'),on=['Mass_Rounded','Retention_Time_Rounded'])
        df2.sort_index(inplace=True)

        df2['Matches'] = np.where((abs(df2['Mass']-df2['Mass_x'])<=0.005) & (abs(df2['Retention_Time']-df2['Retention_Time_x'])<=0.05) & ((df2[a_Abundance[index]].values == df2[b_Abundance[index]].values).any(axis=1)) & (df2['Feature_Number'] != df2['Feature_Number_x']),1,0)
        df2 = df2[df2['Matches']==1]
        df2['N_hits'] = df2[a_Abundance[index]].count(axis=1).round(0)
        df2.sort_values(['N_hits','Mass','Retention_Time'],ascending=[False,False,False],inplace=True)
        #df2['has_Formula'] = np.where( df2['Compound'].str.contains('C') ,1,0)
        #df2['has_not_Formula'] = np.where( df2['Compound'].str.contains('@') ,1,0)
        #df2.to_csv('Testing-duplicates-Algorithm-before.csv', index=False)
        df2 = df2.drop_duplicates(subset=['Compound','Mass','Retention_Time','Feature_Number'], keep="first")
        for i in Abundance:
                df2[i] = df2[i].fillna(df2.groupby(['Compound','Mass_Rounded','Retention_Time_Rounded','Matches'])[i].transform(max))
        #df2.to_csv('Testing-duplicates-Algorithm-after.csv', index=False)
        df2['Group_Position'] = df2.groupby(['Mass_Rounded','Retention_Time_Rounded','Matches']).cumcount()
        df2 = df2[df.columns.values.tolist() + ['Group_Position','N_hits']]
        dft = pd.merge(df,df2,how='left',suffixes=('','_x'),on='Feature_Number')
        dft['Group_Position'] = dft['Group_Position'].fillna(0)
        dft = dft[dft['Group_Position'] == 0]
        dft = dft[df.columns.values.tolist()]
        dft.drop(['Mass_Rounded','Retention_Time_Rounded'],axis=1,inplace=True)
        #dft.to_csv('what_is_in_here.csv',index=False)
        return dft
########### keep looking for solution to failed duplicated  drops & ((df2['Compound'] == df2['Compound_x']) | (df2['Compound'].str.contains('@')))


def MPP_Ready(dft, directory='',file=''):
        dft = dft.rename(columns = {'Compound':'Formula','Retention_Time':'RT'})
        dft['Compound Name'] = dft['Formula']
        dft['CAS ID'] = ""
        Headers = parse_headers(dft,0)
        Abundance = [item for sublist in Headers for item in sublist if len(sublist)>2]
        Blanks = dft.columns[dft.columns.str.contains(pat ='MB_')].tolist()
        Samples = [x for x in Abundance if x not in Blanks]
        NewSamples = common_substrings(Samples)
        for i in range(len(Samples)):
            dft.rename(columns = {Samples[i]:NewSamples[i]},inplace=True)
        #columns = dft.columns.values.tolist()
        #dft = dft.reindex(columns=Columns)
        #print dft
        #dft.to_csv(directory+'/'+file+'_MPP_Ready.csv', index=False)
        dft = dft[['Formula','Compound Name','CAS ID','Mass','RT'] + NewSamples]    
        dft.to_csv(directory+'/'+'Data_Both_Modes_MPP_Ready.csv', index=False)    
        return dft


def Replicates_Number(df,index=0): # calculate Mean,Median,STD,CV for every feature in a sample of multiple replicates
        Headers = [0,0]
        lst = list()
        blanks = ['MB','mb','mB','blank','Blank']                
        Headers[index] = parse_headers(df,index)
        for sub_lst in Headers[index]:
            length = len([s for s in sub_lst if not any(mb in s for mb in blanks)])
            lst.append(length)
        rep = max(lst)
        #df.to_csv('input-updated.csv', index=False)
        return rep




#check_feature_tracers(read_data("House_Dust_Negative_MPP_output.csv"),read_data("Tracers_Table_ for_SRM2585_20170524.csv"))


