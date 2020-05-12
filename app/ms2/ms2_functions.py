#
#


import pandas as pd

from . import scoring
import pymysql as mysql

#  Transforms positive or negative precursor ions to neutral mass. Then searches CFMID database for chemical candidates
#  within a mass error window.
def compare_mgf_df(df_in, mass_error, fragment_error, POSMODE, filtering=False):
    dfg = df_in
    if POSMODE:
        mode = 'ESI-MSMS-pos'
        polarity = ['ESI+', 'Esi+']
        dfg['MASS'] = dfg.groupby('RETENTION TIME')['MASS'].transform(lambda x: (x - 1.0073))
        dfg['MASS'] = dfg['MASS'].round(6)
    else:
        mode = 'ESI-MSMS-neg'
        polarity = ['ESI-', 'Esi-']
        dfg['MASS'] = dfg.groupby('RETENTION TIME')['MASS'].transform(lambda x: (x + 1.0073))
        dfg['MASS'] = dfg['MASS'].round(6)

    mass_list = dfg['MASS'].unique().tolist()
    print(mass_list)
    print("Number of masses to search: " + str(len(mass_list)))
    dfAE_list = list()
    dfS_list = list()
    for mass in mass_list:
        index = mass_list.index(mass) + 1
        print("searching mass " + str(mass) + " number " + str(index) + " of " + str(len(mass_list)))
        dfcfmid = sqlCFMID(mass, mass_error, mode)
        if not dfcfmid:
            print("No matches for this mass in CFMID library, consider changing the accuracy of the queried mass")
        else:
            dfmgf = None
            df = None
            dfmgf = dfg[dfg['MASS'] == mass].reset_index()
            dfmgf.sort_values('MASS', ascending=True, inplace=True)
            df = scoring.Score(dfcfmid, dfmgf, mass, fragment_error, filtering)
            if mode == 'ESI-MSMS-pos':
                df['MASS_in_MGF'] = mass + 1.0073
            if mode == 'ESI-MSMS-neg':
                df['MASS_in_MGF'] = mass - 1.0073

            dfAE_list.append(df)  # all energies scores

    if not dfAE_list:
        print("No matches All Energies found")
    else:
        dfAE_total = pd.concat(dfAE_list)  # all energies scores for all matches
    return dfAE_total
    # dfAE_total.to_excel(filename+'_CFMID_results.xlsx',engine='xlsxwriter')


#  A SQL query to get all the corresponding info from the database
def sqlCFMID(mass=None, mass_error=None, mode=None):

    with open('secrets/secret_nta_db_key_aws.txt') as f:
        pw = f.read().strip()

    # db = mysql.connect(host="mysql-dev1.epa.gov",
    #                    port=3306,
    #                    user='app_nta',
    #                    passwd=pw,
    #                    db="dev_nta_predictions")

    db = mysql.connect(host="qedaurorastack-databaseb269d8bb-1dy6l8bdz01k1.cluster-ro-crqjwmaelnsw.us-east-1.rds.amazonaws.com",
                       port=3306,
                       user='qedadmin',
                       passwd=pw,
                       db="ms2_db")

    cur = db.cursor()
    accuracy_condition = ''
    if mass:
        if mass_error >= 1:
            accuracy_condition = """(abs(c.mass-""" + str(mass) + """)/""" + str(mass) + """)*1000000<""" + str(
                mass_error)
        if mass_error < 1 and mass_error > 0:
            accuracy_condition = """abs(c.mass-""" + str(mass) + """)<=""" + str(mass_error)

    query = """Select t1.dtxcid as DTXCID, t1.formula as FORMULA,t1.mass as MASS, t1.mz as PMASS_x, (t1.intensity/maxintensity)*100.0 as INTENSITY0C,t1.energy as ENERGY 
from 
(select c.dtxcid, max(p.intensity) as maxintensity, p.energy from peak p
Inner Join job j on p.job_id=j.id
Inner Join spectra s on j.spectra_id = s.id
Inner Join chemical c on j.dtxcid=c.dtxcid
#Inner Join fragtool ft on j.fragtool_id=ft.id
#inner join fragintensity fi on fi.peak_id = p.id       
where """ + accuracy_condition + """ 
and s.type='""" + mode + """'
group by c.dtxcid, p.energy) as t2
left join
(select c.*, p.* from peak p
Inner Join job j on p.job_id=j.id
Inner Join spectra s on j.spectra_id = s.id
Inner Join chemical c on j.dtxcid=c.dtxcid
#Inner Join fragtool ft on j.fragtool_id=ft.id   
#inner join fragintensity fi on fi.peak_id = p.id 
where """ + accuracy_condition + """ 
and s.type='""" + mode + """') t1
on t1.dtxcid=t2.dtxcid and t1.energy=t2.energy
order by DTXCID,ENERGY,INTENSITY0C desc;

            """
    print(query)
    # Decided to chunk the query results for speed optimization in post porocessing (spectral matching)
    cur.execute(query)
    chunks = list()
    for chunk in pd.read_sql(query, db, chunksize=1000):
        chunks.append(chunk)
    return chunks
