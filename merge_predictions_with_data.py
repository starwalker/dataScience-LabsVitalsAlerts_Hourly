#This script was written to merge new model predictions into the existing dataframes. When models were continuously being
# developed, I created this script in order to run model simulations on the fly without having to write the predictions
# to a database prior.

import pickle
import pandas as pd
import xgboost as xgb
import numpy as np
import datetime
import gc


key_col = 'PAT_ENC_CSN_ID'
date_col = 'REPORTING_TIME'
bundle_file_name = 'Hourly_Predictions_Bundle.p'

## what are is the model and the prediction column for the model that we would like to use in the merge. Remember - this
# code is used to predict using the new model, and join the predictions to the existing dataframe on the key_col,date_col
pred_col = 'MetMEWS4_24_preds'
model_name = 'MetMEWS4_24'
key_cols = [key_col,date_col]

### create the predictions for the eval dataset (5-1-2018 - 7-31-2018)
#model = pickle.load(open('models/'+model_name+'.p','rb'))
mb = pickle.load(open(bundle_file_name,'rb')).model_dict.get(model_name)
trans = mb.trans
model = mb.model
data = pd.read_pickle('data/raw_dataLV.p')

#the numeric data, found at trans.col_dict.get('zero_imputer_cols') needs to be casted to numeric
data[trans.col_dict.get('zero_imputer_cols')] = data[trans.col_dict.get('zero_imputer_cols')].apply(pd.to_numeric)

features = trans.transform(data)
preds = model.predict(xgb.DMatrix(features,feature_names=trans.feature_names))
preds_df = data.loc[:,[key_col,date_col]]
preds_df[pred_col] = preds

del data
gc.collect()

# sepsis_data =pd.read_pickle('data/sepsis_patient_data.p')
# sepsis_data = sepsis_data[[key_col,date_col,'SIRS_dttm','Blood_Cult_Drawn_dttm','Antibiotic_admin_dttm','Bolus_admin_dttm']]
# # the bolus column ends up being an object(string) that we need to cast to a datetime for comparison
# sepsis_data['Bolus_admin_dttm'] = list(map(lambda x: datetime.datetime.strptime(x, '%m/%d/%y %H:%M') if x not in [None,np.nan] else None,sepsis_data['Bolus_admin_dttm']))
#
#

data = pickle.load(open('data/raw_data.p','rb'))
data = data.drop(pred_col, 1)
data = pd.merge(data, preds_df,  how='left', left_on=[key_col,date_col], right_on = [key_col,date_col])
# data = pd.merge(data, sepsis_data,  how='left', left_on=[key_col,date_col], right_on = [key_col,date_col])
# data['SIRS_dttm_flag24'] = 0
# target_keys = data.loc[np.where(pd.isnull(data['SIRS_dttm'])==False)[0],key_col].unique()
# for key in target_keys:
#     print(key)
#     t0 = data.loc[np.where(data[key_col]==key)[0],'SIRS_dttm'].unique()[0]
#     data.loc[np.where((data[key_col]==key)&(data[date_col]>=t0-np.timedelta64(1, 'D'))&(data[date_col]<t0))[0],'SIRS_dttm_flag24']=1

# del sepsis_data
# gc.collect()

# save the raw_data
file = 'data/raw_data.p'
pickle.dump(data, open(file, 'wb'))




del data
gc.collect()
### create the predictions for the sepsis patient dataset

num_cols = trans.col_dict.get('zero_imputer_cols')
data = pd.read_pickle('data/raw_data_SepsisPatients.p')
data[num_cols] = data[num_cols].apply(pd.to_numeric)
gc.collect()
features = trans.transform(data)


preds = model.predict(xgb.DMatrix(features,feature_names=trans.feature_names))
preds_df = data.loc[:,[key_col,date_col]]
preds_df[pred_col] = preds

del data
gc.collect()

data = pd.read_pickle('data/sepsis_patient_data.p')
# data = data.drop(pred_col, 1)

data = pd.merge(data, preds_df,  how='left', left_on=[key_col,date_col], right_on = [key_col,date_col])
# save the raw_data
file = 'data/sepsis_patient_data.p'
pickle.dump(data, open(file, 'wb'))
