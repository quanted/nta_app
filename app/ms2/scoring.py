import pandas as pd
import numpy as np

LOCAL = True

m=0.5
n=0.5
def Commons(chunks,dfU,fragment_error,filtering):
    print ("starting Commons")
    df_list=list()
    dfL_list=list()
    dfU['MASS_y'] = dfU['MASS'].round(6)
    dfU['MASS'] = dfU['MASS'].round(1)
    dfU['WEIGHTSM'] = (dfU['INTENSITY0M']**m)*(dfU['PMASS_y']**n)
    for chunk in chunks:
        df = None
        dfL = None
        dfInput = None
        dfL = chunk    
        dfL['MASS_x'] = dfL['MASS']
        dfL['MASS'] = dfL['MASS'].round(1)
        dfL['WEIGHTSC'] = (dfL['INTENSITY0C']**m)*(dfL['PMASS_x']**n)
        dfInput = dfU

        df = pd.merge(dfL,dfInput,how='left',on='MASS')             
        if fragment_error >=1:
            df['MATCHES'] = np.where((((abs(df.PMASS_x-df.PMASS_y)/df.PMASS_x)*1000000)<=fragment_error),'1','0') 
        else:
            df['MATCHES'] = np.where((abs(df.PMASS_x-df.PMASS_y)<=fragment_error),'1','0')             
        df.drop(df[df['MATCHES'] == '0'].index,inplace=True)
        df.sort_values(['DTXCID','ENERGY','PMASS_x','INTENSITY0C'],ascending=[True,True,True,False],inplace=True) 
        df_list.append(df)
        dfL_list.append(dfL)

    dft=pd.concat(df_list)
    dfLt=pd.concat(dfL_list)

    # This performs the normalization of peak intensities from raw values into values normalized to maximum intensity observed
    if not LOCAL:
        print("Normalizing intensities")
        dft.rename(columns={'INTENSITY0C': 'old_intensity'}, inplace=True)
        dfLt.rename(columns={'INTENSITY0C': 'old_intensity'}, inplace=True)

        dft_temp = dft.groupby(['MASS', 'ENERGY'], as_index=False)['old_intensity'].max()
        dfLt_temp = dfLt.groupby(['MASS', 'ENERGY'], as_index=False)['old_intensity'].max()
        dft_temp.rename(columns={'old_intensity': 'max_intensity'}, inplace=True)
        dfLt_temp.rename(columns={'old_intensity': 'max_intensity'}, inplace=True)

        dft = pd.merge(dft, dft_temp, how='left', on=['MASS', 'ENERGY'])
        dfLt = pd.merge(dfLt, dfLt_temp, how='left', on=['MASS', 'ENERGY'])

        dft['INTENSITY0C'] = dft['old_intensity'] / dft['max_intensity'] * 100
        dfLt['INTENSITY0C'] = dfLt['old_intensity'] / dfLt['max_intensity'] * 100

        dft.drop('old_intensity', axis=1, inplace=True)
        dft.drop('max_intensity', axis=1, inplace=True)
        dfLt.drop('old_intensity', axis=1, inplace=True)
        dfLt.drop('max_intensity', axis=1, inplace=True)

    if filtering:
        dft.sort_values(['DTXCID','ENERGY','INTENSITY0C'],ascending=[True,True,False],inplace=True) 
        '''select the top 30 matches only and filter out DTXCIDs with less than 5 matches'''
        dft = dft.groupby(['DTXCID','ENERGY']).head(30)
        dft = dft.groupby(['DTXCID','ENERGY']).filter(lambda x: len(x)>=5)

    else:
        dft = dft[(dft['INTENSITY0C']<=100) & (dft['INTENSITY0C']>0.0)]    
    
    WLI = dfLt.groupby(['MASS_x','DTXCID','FORMULA','ENERGY'])['WEIGHTSC'].apply(list).to_dict()    
    WUI = dfU.groupby('MASS_y')['WEIGHTSM'].apply(list).to_dict() 
    WL = dft.groupby(['MASS_x','DTXCID','FORMULA','ENERGY'])['WEIGHTSC'].apply(list).to_dict()
    WU = dft.groupby(['MASS_x','DTXCID','FORMULA','ENERGY'])['WEIGHTSM'].apply(list).to_dict()
    print(len(WL))
    W = list()
    W.append(WL)
    W.append(WU)
    W.append(WLI)
    W.append(WUI)
    return W

def FR(WL,WU):
    num =0.0
    den = 0.0
    SUM = 0.0
    for i in range(0,len(WL)):
        num = WL[i]*WU[i-1]
        den = WL[i-1]*WU[i]
        if (num/den) <= 1:
            l = 1
        else:
            l = -1
        SUM += (num/den)**l     
    F_R = (1.0/float(len(WL)))*SUM
    return F_R

def FD(WL,WU,WLI,WUI):
    #print WL
    #print WU
    SUMU = 0.0
    SUML = 0.0
    SUM = 0.0
    F_D = 0.0
    for i in range(0,len(WUI)):
        SUMU += WUI[i]*WUI[i]
    for i in range(0,len(WLI)):
        SUML += WLI[i]*WLI[i]
    for i in range(0,len(WL)):
        SUM += WL[i]*WU[i]
    F_D = (SUM*SUM)/(SUMU*SUML)
    return F_D    
   
def Score(dfL=None,dfU=None,Mass=0.0,fragment_error=0,filtering=False):
    DF=list()
    W = Commons(dfL,dfU,fragment_error,filtering)
    WL=set(W[0])
    WLI=set(W[2])
    record = list()
    records = list()
    
    for keys in WLI:

        N_LU=0
        F_D=0.0
        F_R=0.0
        score=0.0
        if keys in WLI.intersection(WL):
            N_LU = len(W[0][keys])
            N_U = len(W[3][Mass])            
            F_D = FD(W[0][keys],W[1][keys],W[2][keys],W[3][Mass])
            F_R = FR(W[0][keys],W[1][keys])        
        else:
            F_D = 0.0
        record = list(keys)
        record.append(F_D)
        records.append(record)
    dfi = pd.DataFrame.from_records(records,columns=['MASS','DTXCID','FORMULA','ENERGY','SCORE'])
    df = pd.DataFrame(columns=['MASS','DTXCID','FORMULA','ENERGY','SCORE'])

    if not dfi.empty:
        dfi.sort_values(['ENERGY','SCORE'],ascending=[True,True],inplace=True)
        dfp = pd.pivot_table(dfi,values='SCORE', index='DTXCID',columns='ENERGY').reset_index()
        print(dfp)
        df = pd.merge(dfi,dfp,how='inner',on='DTXCID')
        df.drop(['ENERGY','SCORE'], axis=1,inplace=True)
        df.drop_duplicates(subset=['DTXCID'],keep='first',inplace=True)
        df['energy_sum']= df['energy0'] + df['energy1'] + df['energy2']
        
        df['MASS'] = df['MASS'].apply(lambda x: round(x, 5)) # Get rid of rounding discrepancy
        df.sort_values(['MASS','energy_sum'],ascending=[True,False],inplace=True)    
        df['MATCHES'] = df.groupby(['MASS'])['MASS'].transform('count')

    print ("Number of Matches: " + str(len(WL)))
    return df
