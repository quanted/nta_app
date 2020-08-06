#
#


import pandas as pd
import os
from . import scoring
import psycopg2
import time
import logging
from ...tools.ms2.set_job_status import set_job_status

pw = os.environ.get('AURORA_PW')
LOCAL = False
logger = logging.getLogger("nta_app.ms2")
logger.setLevel(logging.INFO)
print("Test")
print("Test: {}".format(pw))
logger.critical("Test2")

def count_masses(df_in, POSMODE):
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
    return len(mass_list)

#  Transforms positive or negative precursor ions to neutral mass. Then searches CFMID database for chemical candidates
#  within a mass error window.
def compare_mgf_df(df_in, mass_error, fragment_error, POSMODE, mongo, jobid, filtering=False, progress=0):
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
        logger.critical("searching mass " + str(mass) + " number " + str(index) + " of " + str(len(mass_list)))
        t0 = time.clock()
        dfcfmid = sqlCFMID(mass, mass_error, mode)
        t1 = time.clock()
        logger.critical("time for SQL query is :" + str(t1-t0))
        if not dfcfmid:
            logger.critical("No matches for this mass in CFMID library, consider changing the accuracy of the queried mass")
        else:
            dfmgf = None
            df = None
            dfmgf = dfg[dfg['MASS'] == mass].reset_index()
            dfmgf.sort_values('MASS', ascending=True, inplace=True)
            t0 = time.clock()
            df = scoring.Score(dfcfmid, dfmgf, mass, fragment_error, filtering)
            t1 = time.clock()
            logger.critical("time for scoring is :" + str(t1 - t0))
            if mode == 'ESI-MSMS-pos':
                df['MASS_in_MGF'] = mass + 1.0073
            if mode == 'ESI-MSMS-neg':
                df['MASS_in_MGF'] = mass - 1.0073

            dfAE_list.append(df)  # all energies scores
        progress = progress + 1
        set_job_status(mongo, jobid, status='Processing', progress=progress)
        #time.sleep(5)

    if not dfAE_list:
        logger.critical("No matches All Energies found")
        return None
    else:
        dfAE_total = pd.concat(dfAE_list)  # all energies scores for all matches
    return dfAE_total
    # dfAE_total.to_excel(filename+'_CFMID_results.xlsx',engine='xlsxwriter')


#  A SQL query to get all the corresponding info from the database
def sqlCFMID(mass=None, mass_error=None, mode=None):
    pw = os.environ.get('AURORA_PW')
    print("Test")
    print("Test: {}".format(pw))
    logger.critical("Test2")
    if LOCAL:
        logging.critical("LOCAL development mode turned on!")
        import pymysql as mysql
        with open('secrets/secret_nta_db_key.txt') as f:
            pw = f.read().strip()

        db = mysql.connect(host="mysql-dev1.epa.gov",
                           port=3306,
                           user='app_nta',
                           passwd=pw,
                           db="dev_nta_predictions")
    else:
        db = psycopg2.connect(host="qedaurorastack-databaseb269d8bb-1dy6l8bdz01k1.cluster-ro-crqjwmaelnsw.us-east-1.rds.amazonaws.com",
                        port=3306,
                        user='qedadmin',
                        password=pw,
                        database="ms2_db")

    #cur = db.cursor()
    accuracy_condition = ''
    if mass:
        if mass_error >= 1:
            accuracy_condition = """(abs(job_peak.mass-""" + str(mass) + """)/""" + str(mass) + """)*1000000<""" + str(
                mass_error)
        if mass_error < 1 and mass_error > 0:
            accuracy_condition = """abs(job_peak.mass-""" + str(mass) + """)<=""" + str(mass_error)

    query = """select dtxcid as "DTXCID", formula as "FORMULA", mass as "MASS", mz as "PMASS_x", intensity as "INTENSITY0C", energy as "ENERGY"
from job_peak
where """ + accuracy_condition + """ 
and type='""" + mode + """'
order by "DTXCID","ENERGY", "INTENSITY0C" desc;
            """
    if LOCAL:
        if mass:
            if mass_error >= 1:
                accuracy_condition = """(abs(c.mass-""" + str(mass) + """)/""" + str(
                    mass) + """)*1000000<""" + str(
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
    #logger.critical(query)
    #print(query)
    # Decided to chunk the query results for speed optimization in post processing (spectral matching)
    #cur.execute(query)
    chunks = list()
    for chunk in pd.read_sql(query, db, chunksize=1000):
        chunks.append(chunk)
    #cursor.close()
    db.close()
    db = None
    logger.critical("num of chunks: {}".format(len(chunks)))
    #logger.critical("first chunk: {}".format(chunks[0].head()))
    return chunks
