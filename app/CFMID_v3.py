# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:11:13 2017

@author: Hussein Al Ghoul
"""

import pandas as pd
import numpy as np

import CDP_v3 as cdp
import pymysql as mysql
#from sqlalchemy import create_engine
#cnx = create_engine('mysql://root:password@localhost/db')


dfc = None
dfn = None

Adduct_Mass = 1.007825


def parseMGF(file=''):
    NewFile =file.rsplit('.',1)[0] + ".csv"
    with open(file) as f:
        RESULT = list()
        for line in f:
            if line.startswith('TITLE'):
                result = list()
                fields = line.split(' ')
                title, MS, of, pmass, charge, at, RT, mins, delimeter = fields
                #print (pmass, charge, RT)
                #result.append([pmass,RT])
            if line.startswith('RTINSECONDS'):
                RTS = line.split('=')[1]
                for line in f:
                    if line.split(' ')[0] == 'END':
                        break
                    a, b  = line.split('\t')
                    result.append([float(pmass), float(RT), charge, float(a),float(b)])
                RESULT.append(result)
        #print RESULT[0]
    categories = [ "RUN %s" %i for i in range(0,len(RESULT))]
    dfg = pd.concat([pd.DataFrame(d) for d in RESULT], keys=categories)
    dfg.columns = ["MASS", "RETENTION TIME", "CHARGE", "PMASS_y","INTENSITY"]
    dfg.sort_values(['MASS','RETENTION TIME'],ascending=[True,True],inplace=True) 
    df1 = dfg.reset_index()
    df1['TOTAL INTENSITY'] = df1.groupby(['MASS','RETENTION TIME'])['INTENSITY'].transform(sum)
    df1.sort_values(['MASS','TOTAL INTENSITY'],ascending=[True,True],inplace=True)
    df1 = df1.groupby('MASS').apply(lambda x: x[x['TOTAL INTENSITY'] == x['TOTAL INTENSITY'].max()])
    df1.loc[df1['PMASS_y']>df1['MASS'],'INTENSITY']=None
    df1.dropna(inplace=True)
    df1.sort_values(['MASS','INTENSITY'],ascending=[True,False],inplace=True)
    #df1['INTENSITY0M'] = df1.groupby('MASS')['INTENSITY'].apply(lambda x: (x/x.nlargest(2).min())*100.0)
    df1['INTENSITY0M'] = (df1['INTENSITY']/df1.groupby('MASS')['INTENSITY'].transform(np.max))*100.0
    df1.loc[df1['INTENSITY0M']>100,'INTENSITY0M']=None
    #df1.loc[df1['INTENSITY0M']<0.1,'INTENSITY0M']=None
    df1.reset_index(drop=True, inplace=True) # reset index
    df1.to_csv(NewFile,float_format='%.5f',index=False)
    dfg = df1
    #dfg.to_csv("CE10d_mgf.csv",index=False)
    return dfg


def spectrum_reader(file=''):
    dfg = pd.read_csv(file)
    #dfg = dfg.groupby(['MASS','RETENTION TIME']).head(30)
    dfg = dfg[(dfg['INTENSITY0M']<=100) & (dfg['INTENSITY0M']>0.0)]
    return dfg


''' A SQL query to get all the corresponding info from the database'''
def sqlCFMID(mass,ppm,mode,formula=None):
    db = mysql.connect(host="mysql-dev1.epa.gov",
                   user="halghoul",
                   passwd="P@ssw0rd",
                   db="sbox_tcathe02_mspredict")
    cur = db.cursor()
    accuracy_condition = ''
    if ppm>=1.0:
        accuracy_condition = """(abs(c.mass-"""+str(mass)+""")/"""+str(mass)+""")*1000000<"""+str(ppm)
    if ppm<1.0 and ppm>0: 
        accuracy_condition = """(abs(c.mass-"""+str(mass)+""")"""
    if formula:
        accuracy_condition = """c.formula='"""+formula+"""'"""
    query= """select t1.dtxcid as DTXCID, t1.formula as FORMULA,t1.mass as MASS, t1.mz as PMASS_x, (t1.intensity/maxintensity)*100.0 as INTENSITY0C,
t1.energy as ENERGY 
from 
	(select c.*, p.* from peak p
		Inner Join job j on p.job_id=j.id
		Inner Join chemical c on j.dtxcid=c.dtxcid
		Inner Join spectra s on j.spectra_id = s.id
		Inner Join fragtool ft on j.fragtool_id=ft.id        
		where """ +accuracy_condition + """ 
       and s.type='""" + mode + """') t1
left JOIN (select c.dtxcid, max(p.intensity) as maxintensity, p.energy from peak p
			Inner Join job j on p.job_id=j.id
			Inner Join chemical c on j.dtxcid=c.dtxcid
			Inner Join spectra s on j.spectra_id = s.id
			Inner Join fragtool ft on j.fragtool_id=ft.id
		    where """ +accuracy_condition + """ 
       and s.type='""" + mode + """'
			group by c.dtxcid, p.energy) t2
on t1.dtxcid=t2.dtxcid and t1.energy=t2.energy
order by DTXCID,ENERGY,INTENSITY0C desc;"""
    #Decided to chunk the query results for speed optimization in post porocessing (spectral matching)
    cur.execute(query)
    chunks=list()
    for chunk in pd.read_sql(query,db,chunksize=1000):
        chunks.append(chunk)
    return chunks
            

def list_maker(dfpc,dfg,mode):
    # make a list of the MGF masses corresponding to the NTA Monoisotopic masses 
    dfg['Mass_rounded'] = dfg['MASS'].round(1)
    dfpc = dfpc[['Mass','Ionization_Mode']]
    dfpcdl = dfpc.loc[dfpc['Ionization_Mode'] == mode]
    dfpcdl['Mass_rounded'] = dfpcdl['Mass'].round(1)
    df = pd.merge(dfpcdl,dfg,how='left',on='Mass_rounded') 
    df['MATCHES'] = np.where((((abs(df['Mass']-df['MASS'])/df['MASS'])*1000000)<=15),'1','0') 
    df.drop(df[df['MATCHES'] == '0'].index,inplace=True)
    df=df[np.isfinite(df['MASS'])]
    mlist = df['MASS'].unique().tolist()
    print(mlist)
    return mlist

    


def compare_mgf_df(file,dfpc,ppm,ppm_sl,POSMODE,filtering):
    dfg = spectrum_reader(file)
    if POSMODE:
        mode='ESI-MSMS-pos'
        polarity=['ESI+','Esi+']
        #CMass = Mass - Adduct_Mass
        dfg['MASS'] = dfg.groupby('RETENTION TIME')['MASS'].transform(lambda x: (x-1.007825))
        dfg['MASS'] = dfg['MASS'].round(6)
    else:
        mode='ESI-MSMS-neg'
        polarity=['ESI-','Esi-']
        #CMass = Mass + Adduct_Mass
        dfg['MASS'] = dfg.groupby('RETENTION TIME')['MASS'].transform(lambda x: (x+1.007825)) 
        dfg['MASS'] = dfg['MASS'].round(6)
    #dfg.to_csv("dfg_aftercsv.csv",float_format='%.7f',index=False)  
    #mass_list = dfg['MASS'].unique().tolist()
    #mass_list = [312.184525]
    print(dfpc)
    Polarity = polarity[0]
    mass_list = list_maker(dfpc,dfg,Polarity)
    if not mass_list:
        Polarity = polarity[1]
        mass_list = list_maker(dfpc,dfg,Polarity)
    print(mass_list)
    print(("Number of masses to search: " + str(len(mass_list))))
    dfAE_list=list()
    dfS_list=list()  
    for mass in mass_list:
        index = mass_list.index(mass) + 1
        print(("searching mass " + str(mass) + " number " + str(index) + " of " + str(len(mass_list))))
        dfcfmid = sqlCFMID(mass,ppm,mode)
        if not dfcfmid:
            print("No matches for this mass in CFMID library, consider changing the accuracy of the queried mass")
        else:    
            dfmgf = None
            df = None
            dfmgf = dfg[dfg['MASS'] == mass].reset_index()
            dfmgf.sort_values('MASS',ascending=True,inplace=True)
            df = cdp.Score(dfcfmid,dfmgf,mass,ppm_sl,filtering)
            if mode=='ESI-MSMS-pos':
                df[0]['MASS_in_MGF'] = mass + 1.007825
                df[1]['MASS_in_MGF'] = mass + 1.007825
            if mode=='ESI-MSMS-neg':
                df[0]['MASS_in_MGF'] = mass - 1.007825
                df[1]['MASS_in_MGF'] = mass - 1.007825
            dfAE_list.append(df[0]) #all energies scores
            dfS_list.append(df[1]) #sum of all energies score
            
    dfAE_total = pd.concat(dfAE_list) #all energies scores for all matches
    dfS_total = pd.concat(dfS_list) #Sum of scores for all matches
    dfAE_total['Ionization_Mode'] = Polarity
    dfS_total['Ionization_Mode'] = Polarity    
    #th_dtxcid = dfAE_list[0].at[0,'DTXCID'] #top hit dtxcid for plotting
    #print(th_dtxcid) 
    
    df_Result = [dfAE_total,dfS_total]
    return df_Result

    #dfcfm = pd.concat(dfcfmid)[(pd.concat(dfcfmid)['DTXCID'] == th_dtxcid) & (pd.concat(dfcfmid)['ENERGY'] == 'energy2')].reset_index()
    #cdp.plot(dfcfm,dfmgf)
    '''
    df_resultAE = merge_pcdl(dfpc,dfAE_total,Polarity)
    if POSMODE:
        AEFile = "Pos_Mode_compared_with_CFMID_MultiScores_wformula_DTXCID.xlsx"
    else:
        AEFile = "Neg_Mode_compared_with_CFMID_MultiScores_wformula_DTXCID.xlsx"
        
    df_resultAE.to_excel(AEFile,na_rep='',engine='xlsxwriter')
    
    df_resultS = merge_pcdl(dfpc,dfS_total,Polarity)
    if POSMODE:
        File = "Pos_Mode_compared_with_CFMID_OneScore_wformula_DTXCID.xlsx"
    else:
        File = "Neg_Mode_compared_with_CFMID_OneScore_wformula_DTXCID.xlsx"        
    df_resultS.to_excel(File,na_rep='',engine='xlsxwriter')  
    return dfAE_total
    '''


def merge_pcdl(dfpcdl,df):
    #df['Ionization_Mode'] = polarity
    #print df
    dfpcdl['MPP_ASSIGNED_FORMULA'] = dfpcdl['MPP_ASSIGNED_FORMULA'].str.replace(' ','')
    dfm = pd.merge(dfpcdl,df,how='left',left_on=['DTXCID_INDIVIDUAL_COMPONENT','Ionization_Mode'],right_on=['DTXCID','Ionization_Mode'])
    dfm.fillna('',inplace=True)
    print(dfpcdl.columns.values.tolist()[:64])
    #dfm = dfm.set_index(dfpcdl.columns.values.tolist()[:64])
    #dfm.sort_values(['Neutral Monoisotopic Mass','MASS','RANK'],ascending=[True,True,True],inplace=True)
    return dfm



def indexing(file):
    df = pd.read_csv(file)
    df.set_index(df.columns.values.tolist(),inplace=True)
    print(df)
    df.to_excel('try_indexing_outcome.xlsx',engine='xlsxwriter')


        


    
#read_NTA_data('/home/hussein/Documents/NTA/Python_alt/ENTACT_DataReporting_EPA_MS2.csv')
#parseCFMID('/home/hussein/Documents/NTA/Python_alt/spectra_ESI-MSMS-neg_mass.dat')
#compare_df(183.057312)
#compare_df(183.058217) 
    
#parseMGF(os.getcwd()+'/20180418_505_CE40.mgf') #<--Convert MGF to CSV

# to read a signle spectrum from text file input spectrum:
#input_parser('input_spectrum.txt') 


# to process MGD files uncomment the following lines
#indexing('try_indexing.csv')
'''
file = os.getcwd()+'/20180418_505_CE40.csv'
fpcdl = os.getcwd()+'/20180419 505 pos CE40_PCDL.csv'
t0=time.clock()
compare_mgf_df(file,fpcdl,10,0.02,POSMODE=True)
t1=time.clock()
print ("time to Process is " + str(t1-t0))
'''








