# This is to query all necessary data for this project

if __name__ == '__main__':
    import pickle
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    import datetime
    from TextMiningMachine.io import get_data
    from TextMiningMachine.io import read_sql_text
    import gc

    # Load the query
    query = read_sql_text('sql_scripts/query_data.sql')

    # querry the data
    data = get_data('muscedw', query)

    # save the transform
    file = 'data/raw_data.p'
    pickle.dump(data, open(file, 'wb'))


## pull data for sepsis patients
    # Load the query
    query = read_sql_text('sql_scripts/query_sepsis_patients.sql')

    # querry the data
    data = get_data('muscedw', query)

    # save the transform
    file = 'data/sepsis_patient_data.p'
    pickle.dump(data, open(file, 'wb'))


## pull data for raw data
    # Load the query
    query = read_sql_text('sql_scripts/query_raw_data.sql')

    # querry the data
    data = get_data('muscedw', query)

    # save the transform
    file = 'data/raw_dataLV.p'
    pickle.dump(data, open(file, 'wb'))

