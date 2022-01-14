import pandas as pd
import os
from ..feature.feature import *
import psycopg2
import time
import logging
from ...tools.ms2.set_job_status import set_job_status
from .utilities import *

pw = os.environ.get('AURORA_PW')
LOCAL = False
logger = logging.getLogger("nta_app.ms2")
logger.setLevel(logging.INFO)

#  A SQL query to get all the corresponding info from the database
def sqlCFMID(mass=None, mass_error=None, mode=None):
    pw = os.environ.get('AURORA_PW')
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
            max_mass = mass + mass * mass_error / 1000000
            min_mass = mass - mass * mass_error / 1000000
            accuracy_condition = """job_peak.mass BETWEEN """ + str(min_mass) + """ AND """ + str(max_mass)
        if mass_error < 1 and mass_error > 0:
            max_mass = mass + mass_error
            min_mass = mass - mass_error
            accuracy_condition = """job_peak.mass BETWEEN """ + str(min_mass) + """ AND """ + str(max_mass)

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
    #chunks = list()
    #for chunk in pd.read_sql(query, db, chunksize=1000):
    #    chunks.append(chunk)
    sql_df = pd.read_sql(query, db)
    #cursor.close()
    db.close()
    db = None
    sql_df.rename(columns = {'PMASS_x':'FRAG_MASS', 'INTENSITY0C':'FRAG_INTENSITY'}, inplace = True)
    formated_data_dict = sql_df.groupby(['DTXCID', 'FORMULA', 'MASS'])[['ENERGY','FRAG_MASS','FRAG_INTENSITY']]\
        .apply(lambda x: x.groupby('ENERGY')[['FRAG_MASS','FRAG_INTENSITY']]\
               .apply(lambda x: x.to_dict('list'))).to_dict('index')
    #logger.critical('Num of chunks: {}'.format(len(cfmid_chunk_list)))
    return formated_data_dict

