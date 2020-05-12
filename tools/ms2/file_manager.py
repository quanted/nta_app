
from __future__ import absolute_import
import pandas as pd
import numpy as np
import io
import csv

def parse_mgf(file_in):
    #NFile = file.rsplit('/', 1)[-1]
    #NewFile = NFile.rsplit('.', 1)[0] + ".csv"
    #with open(file) as f:
    decoded_file = file_in.read().decode('utf-8')
    io_string = io.StringIO(decoded_file)
    with io_string as f:
    #print(file)
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
                    a, b = line.split('\t')
                    result.append([float(pmass), float(RT), charge, float(a), float(b)])
                RESULT.append(result)
    categories = ["RUN %s" % i for i in range(0, len(RESULT))]
    dfg = pd.concat([pd.DataFrame(d) for d in RESULT], keys=categories)
    dfg.columns = ["MASS", "RETENTION TIME", "CHARGE", "PMASS_y", "INTENSITY"]
    dfg.sort_values(['MASS', 'RETENTION TIME'], ascending=[True, True], inplace=True)
    df1 = dfg.reset_index()
    df1['TOTAL INTENSITY'] = df1.groupby(['MASS', 'RETENTION TIME'])['INTENSITY'].transform(sum)
    df1.sort_values(['MASS', 'TOTAL INTENSITY'], ascending=[True, True], inplace=True)
    df1 = df1.groupby('MASS').apply(lambda x: x[x['TOTAL INTENSITY'] == x['TOTAL INTENSITY'].max()])
    df1.loc[df1['PMASS_y'] > df1['MASS'], 'INTENSITY'] = None
    df1.dropna(inplace=True)

    df1 = df1.rename(
        columns={'MASS': 'MASS2'})  # Had to rename mass column because of some ambiguity with the name MASS
    df1.sort_values(['MASS2', 'INTENSITY'], ascending=[True, False], inplace=True)

    df1['INTENSITY0M'] = (df1['INTENSITY'] / df1.groupby('MASS2')['INTENSITY'].transform(np.max)) * 100.0

    df1 = df1.rename(columns={'MASS2': 'MASS'})  # Rename MASS2 back to original name

    df1.loc[df1['INTENSITY0M'] > 100, 'INTENSITY0M'] = None
    df1.reset_index(drop=True, inplace=True)  # reset index
    #df1.to_csv(NewFile, float_format='%.5f', index=False)
    print(df1)
    return df1
