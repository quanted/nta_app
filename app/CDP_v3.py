# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 09:57:30 2018

@author: HALGhoul
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 10:11:13 2017

@author: Hussein Al Ghoul
"""

import pandas as pd
import numpy as np

m=0.5
n=0.5
def Commons(chunks,dfU,ppm_sl,filtering,energy=False):
    print ("starting Commons")
    df_list=list()
    dfL_list=list()
    dfU['MASS_y'] = dfU['MASS'].round(6) 
    #dfU.rename(columns={'PMASS':'PMASS_y'},inplace=True) 
    dfU['MASS'] = dfU['MASS'].round(1)
    dfU['WEIGHTSM'] = (dfU['INTENSITY0M']**m)*(dfU['PMASS_y']**n)
    for chunk in chunks:
        df = None
        dfL = None
        dfInput = None
        dfL = chunk    
        dfL['MASS_x'] = dfL['MASS']
        #dfL.rename(columns={'PMASS':'PMASS_x'},inplace=True)   
        dfL['MASS'] = dfL['MASS'].round(1)
        #dfL = dfL.groupby(['MASS','DTXCID']).head(30)
        dfL['WEIGHTSC'] = (dfL['INTENSITY0C']**m)*(dfL['PMASS_x']**n)
        dfInput = dfU
        if energy:
            df = pd.merge(dfL,dfInput,how='left',on=['MASS','ENERGY']) 
        else:
            df = pd.merge(dfL,dfInput,how='left',on='MASS')             
        if ppm_sl >=1:
            df['MATCHES'] = np.where((((abs(df.PMASS_x-df.PMASS_y)/df.PMASS_x)*1000000)<=ppm_sl),'1','0') 
        else:
            df['MATCHES'] = np.where((abs(df.PMASS_x-df.PMASS_y)<=ppm_sl),'1','0')             
        df.drop(df[df['MATCHES'] == '0'].index,inplace=True)
        df.sort_values(['DTXCID','ENERGY','PMASS_x','INTENSITY0C'],ascending=[True,True,True,False],inplace=True) 
        df_list.append(df)
        dfL_list.append(dfL)

    dft=pd.concat(df_list)
    dfLt=pd.concat(dfL_list)
    #dft.to_csv("cfmid_match.csv",index=False)

    if filtering:
        #dft['INTENSITY0CR'] = (dft['INTENSITY0C']/dft.groupby(['DTXCID','ENERGY'])['INTENSITY0C'].transform(sum))
        #dft.sort_values(['DTXCID','ENERGY','INTENSITY0C'],ascending=[True,True,False],inplace=True) 
        #dft['cumsum'] = dft['INTENSITY0CR'].cumsum()
        #dfLt = dfLt[dfLt['INTENSITY0CR'] > dfLt['INTENSITY0CR'].quantile(0.3)]
        '''select the top 30 matches only and filter out DTXCIDs with less than 5 matches'''
        dfLt = dfLt.groupby(['DTXCID','ENERGY']).head(30)
        dfLt = dfLt.groupby(['DTXCID','ENERGY']).filter(lambda x: len(x)>=5)
        #print dfLt.groupby(['DTXCID','ENERGY']).filter(lambda s: s.INTENSITY0CR.sum() <= 0.8)
        #dfLt = dfLt[dfLt['cumsum']<=0.8]
        #print dfL[dfL['INTENSITY0CR'].groupby(['DTXCID','ENERGY']).transform('sum') <=0.8]
        #print dfL
    else:
        dfLt = dfLt[(dfLt['INTENSITY0C']<=100) & (dfLt['INTENSITY0C']>0.0)]    
    
    #dfLt.to_csv("cfmid.csv",index=False)

    WLI = dfLt.groupby(['MASS_x','DTXCID','FORMULA','ENERGY'])['WEIGHTSC'].apply(list).to_dict()  
    #print WLI    
    WUI = dfU.groupby('MASS_y')['WEIGHTSM'].apply(list).to_dict() 
    #df.to_csv("Commons_Output.csv",index=False)
    WL = dft.groupby(['MASS_x','DTXCID','FORMULA','ENERGY'])['WEIGHTSC'].apply(list).to_dict()
    WU = dft.groupby(['MASS_x','DTXCID','FORMULA','ENERGY'])['WEIGHTSM'].apply(list).to_dict()
    print((len(WL)))
    #print WUI
    W = list()
    W.append(WL)
    W.append(WU)
    W.append(WLI)
    W.append(WUI)
    return W



def FD(WL,WU,WLI,WUI):
    #print WL
    #print WU
    SUMU = 0.0
    SUML = 0.0
    SUM = 0.0
    F_D = 0.0
    #print WUI
    for i in range(0,len(WUI)):
        #print WUI[i]
        SUMU += WUI[i]*WUI[i]
    #print SUMU
    for i in range(0,len(WLI)):
        SUML += WLI[i]*WLI[i]
    #print SUML
    for i in range(0,len(WL)):
        #print WU[i]
        SUM += WL[i]*WU[i]
    #print SUM
    F_D = (SUM*SUM)/(SUMU*SUML)
    #print F_D
    return F_D    
   
def Score(dfL=None,dfU=None,Mass=0.0,ppm_sl=0,filtering=False,energy=False):
    DF=list()
    W = Commons(dfL,dfU,ppm_sl,filtering,energy)
    WL=set(W[0])
    #print WL
    WLI=set(W[2])
    record = list()
    records = list()
    #print WLI
    for keys in WLI.intersection(WLI):
        if keys in WLI.intersection(WL):
            #print W[0][keys] 
            #print W[1][Mass]
            F_D=0.0
            F_D = FD(W[0][keys],W[1][keys],W[2][keys],W[3][Mass])
        else:
            F_D = 0.0
            #score = ((N_U*F_D) + (N_LU*F_R))/(N_U + N_LU)
        record = list(keys)
        record.append(F_D)
        #record.append(score)
        records.append(record)
    #dfL_plot = dfL.loc[dfL['DTXCID'].isin([keys[1]])].reset_index()
    #plot(dfL_plot,dfU)
    dfi = pd.DataFrame.from_records(records,columns=['MASS','DTXCID','FORMULA','ENERGY','SCORE'])
    df = pd.DataFrame(columns=['MASS','DTXCID','FORMULA','ENERGY','SCORE'])
    dfs = pd.DataFrame(columns=['MASS','DTXCID','FORMULA','ENERGY','SCORE'])
    if not dfi.empty:
        dfi.sort_values(['ENERGY','SCORE'],ascending=[True,True],inplace=True)
        #df['RANK'] = df.groupby(['FORMULA','ENERGY'])['SCORE'].rank(method='dense',ascending=False) 
        #df = df.pivot(index='DTXCID', columns='ENERGY', values='SCORE')
        dfp = pd.pivot_table(dfi,values='SCORE', index='DTXCID',columns='ENERGY').reset_index()
        print(dfp)
        df = pd.merge(dfi,dfp,how='inner',on='DTXCID')
        df.drop(['ENERGY','SCORE'], axis=1,inplace=True)
        df.drop_duplicates(subset=['DTXCID'],keep='first',inplace=True)
        if 'energy0' in df:
            df['RANK_E0'] = df.groupby(['FORMULA'])['energy0'].rank(method='dense',ascending=False) 
        else:
            df['RANK_E0'] = None
        if 'energy1' in df:    
            df['RANK_E1'] = df.groupby(['FORMULA'])['energy1'].rank(method='dense',ascending=False) 
        else:
            df['Rank_E1'] = None
        if 'energy2' in df: 
            df['RANK_E2'] = df.groupby(['FORMULA'])['energy2'].rank(method='dense',ascending=False)
        else:
            df['Rank_E2'] = None
        df.sort_values(['MASS','FORMULA','RANK_E0'],ascending=True,inplace=True)    
        df['MATCHES'] = df.groupby(['MASS','FORMULA'])['FORMULA'].transform('count')
        df.reset_index()
        #df.to_csv('try_the_index.csv',index=False)
        print (df)
        #df.to_csv('Score_Alllevels.csv',index=False)
        dfs = dfi.groupby(['DTXCID','MASS','FORMULA'],as_index=False)['SCORE'].sum()
        dfs.reset_index()
        dfs['RANK'] = dfs.groupby(['FORMULA'])['SCORE'].rank(method='dense',ascending=False) # rank according to formula here by adding ['FORMULA','ENERGY']
        dfs['MATCHES'] = dfs.groupby(['MASS','FORMULA'])['FORMULA'].transform('count')
        dfs.sort_values(['MASS','FORMULA','RANK'],ascending=True,inplace=True)    
    #print (dfs)
    #dfs.to_csv("Score_Sum.csv",index=False)
    print(("Number of Matches: " + str(len(WL))))
    DF.append(df)
    DF.append(dfs)
    return DF


















    
    
    
    
