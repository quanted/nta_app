# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:11:13 2017

@author: Hussein Al Ghoul

NOTE: Update the user and password in the sql funstion to access the cfmid server
only the first 20 masses will be tested in this code, if you want to run the full file change     
    for mass in mass_list[0:20]:
        to
    for mass in mass_list:    

"""

import os
import pandas as pd
import numpy as np

import CosineDotProduct_v24 as cpd
import pymysql as mysql



def parseMGF(file=''):
    NFile = file.rsplit('/',1)[-1]
    NewFile = NFile.rsplit('.',1)[0] + ".csv"
    with open(file) as f:
        RESULT = list()
        for line in f:
            if line.startswith('TITLE'):
                result = list()
                fields = line.split(' ')
                title, MS, of, pmass, charge, at, RT, mins, delimeter = fields

            if line.startswith('RTINSECONDS'):
                RTS = line.split('=')[1]
                for line in f:
                    if line.split(' ')[0] == 'END':
                        break
                    a, b  = line.split('\t')
                    result.append([float(pmass), float(RT), charge, float(a),float(b)])
                RESULT.append(result)
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
    
    # AC 20190620 edited due to error: "ValueError: 'MASS' is both an index level and a column label, which is ambiguous."
    df1 = df1.rename(columns={'MASS':'MASS2'}) # Had to rename mass column because of some ambiguity with the name MASS
    df1.sort_values(['MASS2','INTENSITY'],ascending=[True,False],inplace=True) 

    df1['INTENSITY0M'] = (df1['INTENSITY']/df1.groupby('MASS2')['INTENSITY'].transform(np.max))*100.0
    
    df1 = df1.rename(columns={'MASS2':'MASS'}) # Rename MASS2 back to original name
    
    df1.loc[df1['INTENSITY0M']>100,'INTENSITY0M']=None
    df1.reset_index(drop=True, inplace=True) # reset index
    df1.to_csv(NewFile,float_format='%.5f',index=False)
    dfg = df1
    return dfg



#def spectrum_reader(file=''):
#    dfg = pd.read_csv(file)
#    dfg = dfg[(dfg['INTENSITY0M']<=100) & (dfg['INTENSITY0M']>0.0)]
#    return dfg


''' A SQL query to get all the corresponding info from the database'''
def sqlCFMID(mass=None,mass_error=None,mode=None):
    db = mysql.connect(host="mysql-dev1.epa.gov",
                    port=3306,
                   user='app_nta',
                   passwd=None,
                   db="dev_nta_predictions")     
                         
    cur = db.cursor()
    accuracy_condition = ''
    if mass:
        if mass_error>=1:
            accuracy_condition = """(abs(c.mass-"""+str(mass)+""")/"""+str(mass)+""")*1000000<"""+str(mass_error)
        if mass_error<1 and mass_error>0: 
            accuracy_condition = """abs(c.mass-"""+str(mass)+""")<="""+str(mass_error)

    query= """Select t1.dtxcid as DTXCID, t1.formula as FORMULA,t1.mass as MASS, t1.mz as PMASS_x, (t1.intensity/maxintensity)*100.0 as INTENSITY0C,t1.energy as ENERGY 
from 
(select c.dtxcid, max(p.intensity) as maxintensity, p.energy from peak p
Inner Join job j on p.job_id=j.id
Inner Join spectra s on j.spectra_id = s.id
Inner Join chemical c on j.dtxcid=c.dtxcid
#Inner Join fragtool ft on j.fragtool_id=ft.id
#inner join fragintensity fi on fi.peak_id = p.id       
where """ +accuracy_condition + """ 
and s.type='""" + mode + """'
group by c.dtxcid, p.energy) as t2
left join
(select c.*, p.* from peak p
Inner Join job j on p.job_id=j.id
Inner Join spectra s on j.spectra_id = s.id
Inner Join chemical c on j.dtxcid=c.dtxcid
#Inner Join fragtool ft on j.fragtool_id=ft.id   
#inner join fragintensity fi on fi.peak_id = p.id 
where """ +accuracy_condition + """ 
and s.type='""" + mode + """') t1
on t1.dtxcid=t2.dtxcid and t1.energy=t2.energy
order by DTXCID,ENERGY,INTENSITY0C desc;
          
            """
    print(query)
    #Decided to chunk the query results for speed optimization in post porocessing (spectral matching)
    cur.execute(query)
    chunks=list()
    for chunk in pd.read_sql(query,db,chunksize=1000):
        chunks.append(chunk)
    return chunks
            

#  Transforms positive or negative precursor ions to neutral mass. Then searches CFMID database for chemical candidates
#  within a mass error window.
def compare_mgf_df(df_in, mass_error, fragment_error, POSMODE, filtering=False):
    dfg = df_in
    if POSMODE:
        mode='ESI-MSMS-pos'
        polarity=['ESI+','Esi+']
        dfg['MASS'] = dfg.groupby('RETENTION TIME')['MASS'].transform(lambda x: (x-1.0073))
        dfg['MASS'] = dfg['MASS'].round(6)
    else:
        mode='ESI-MSMS-neg'
        polarity=['ESI-','Esi-']
        dfg['MASS'] = dfg.groupby('RETENTION TIME')['MASS'].transform(lambda x: (x+1.0073)) 
        dfg['MASS'] = dfg['MASS'].round(6)

    mass_list = dfg['MASS'].unique().tolist()        
    print(mass_list)
    print("Number of masses to search: " + str(len(mass_list)))
    dfAE_list=list()
    dfS_list=list()  
    for mass in mass_list:
        index = mass_list.index(mass) + 1
        print("searching mass " + str(mass) + " number " + str(index) + " of " + str(len(mass_list)))
        dfcfmid = sqlCFMID(mass,mass_error,mode)
        if not dfcfmid:
            print("No matches for this mass in CFMID library, consider changing the accuracy of the queried mass")
        else:    
            dfmgf = None
            df = None
            dfmgf = dfg[dfg['MASS'] == mass].reset_index()
            dfmgf.sort_values('MASS',ascending=True,inplace=True)
            df = cpd.Score(dfcfmid,dfmgf,mass,fragment_error,filtering)
            if mode=='ESI-MSMS-pos':
                df['MASS_in_MGF'] = mass + 1.0073
            if mode=='ESI-MSMS-neg':
                df['MASS_in_MGF'] = mass - 1.0073


            dfAE_list.append(df) #all energies scores

    if not dfAE_list:
        print("No matches All Energies found")
    else:
        dfAE_total = pd.concat(dfAE_list) #all energies scores for all matches
    return dfAE_total
    #dfAE_total.to_excel(filename+'_CFMID_results.xlsx',engine='xlsxwriter')


