from TextMiningMachine.io import get_data
from TextMiningMachine.feature_extraction import DataSetBuilder
from TextMiningMachine.xgboost_optimizer import XgboostOptimizer
import pandas as pd
import pickle
import xgboost as xgb
from nltk.corpus import stopwords
import numpy as np
from datetime import timedelta
import datetime
import os
import gc
from scipy import sparse

data = pd.read_pickle('data/raw_data.p')
print(data.shape[0])
trans = pickle.load(open('dsvm_build_files/text_cat_transformer.p','rb'))
keep_cols = ['PAT_ID', 'PAT_ENC_CSN_ID', 'REPORTING_TIME']
#target_cols = ['MetSIRS4_4hr_8', 'MetSIRS4_4hr_24', 'MetSIRS4_4hr_48']

key_col = 'PAT_ENC_CSN_ID'
date_col = 'REPORTING_TIME'
target_col = 'SIRS_dttm_flag24'

sepsis_data =pd.read_pickle('data/sepsis_patient_data.p')
sepsis_data = sepsis_data[[key_col,date_col,'SIRS_dttm','Blood_Cult_Drawn_dttm','Antibiotic_admin_dttm','Bolus_admin_dttm']]
# the bolus column ends up being an object(string) that we need to cast to a datetime for comparison
sepsis_data['Bolus_admin_dttm'] = list(map(lambda x: datetime.datetime.strptime(x, '%m/%d/%y %H:%M') if x not in [None,np.nan] else None,sepsis_data['Bolus_admin_dttm']))
data = pd.merge(data, sepsis_data,  how='left', left_on=[key_col,date_col], right_on = [key_col,date_col])
target_keys = data.loc[np.where(pd.isnull(data['SIRS_dttm'])==False)[0],key_col].unique()

del sepsis_data
gc.collect()


#censored_inds = []
data[target_col] = 0
dttm_inds = []
for key in target_keys:
    print(key)
    key_inds = data[key_col]==key
    t0 = data.loc[np.where(key_inds)[0],'SIRS_dttm'].unique()[0]
    dttm_inds.extend(np.where((key_inds)&(data[date_col]<=t0)&(data[date_col]>=t0-np.timedelta64(1, 'D')))[0])

data.loc[dttm_inds,target_col]=1
data[target_col] = list(map(lambda x: 1 if x>0 else 0, (data[target_col]+data['MetSIRS4_4hr_24']).values))

weights = np.ones((data.shape[0],1))
weights[np.where((data['SIRS4_4hr_Countdown']<=24) & (data['SIRS4_4hr_Countdown']>0))[0]] = 6
weights[dttm_inds] = 6
#censored_inds.extend(np.where((data[key_col]==key)&(data[date_col]>t0))[0])
cant_convert_cols = []
for col in trans.col_dict.get('zero_imputer_cols'):
    if data[col].dtype!=float:
        print(col)
        try:
            data[col] = data[col].astype(float)
        except:
            cant_convert_cols.append(col)



# features = sparse.load_npz('data/features.npz')
# features = sparse.csr_matrix(features)
num_rows = data.shape[0]
rows_per_iter = 8000000

bounds = list(range(0,num_rows,rows_per_iter))
bounds.append(num_rows)

for i in range((len(bounds)-1)):
    print('transforming, iteration '+str(i)+' of '+str(len(bounds)-1))
    lb = bounds[i]
    ub = bounds[i+1]
    if i ==0:
        features = trans.transform(data.iloc[lb:ub,:])
    else:
        features = sparse.vstack([features,trans.transform(data.iloc[lb:ub,:])])
# print(features.shape)


#
#
# keep_inds =np.where((data['NoSIRS4_4hr_4']==1) | (data['MetSIRS4_4hr']==1))[0]
# data = data.iloc[keep_inds,:].reset_index(drop = True)
# features = features[keep_inds,:]


#split the date into 70%, 20%, 10% for train, test, eval resp.
train_cut_date = data[date_col].sort_values().reset_index(drop=True)[int(data.shape[0] *.60)]
test_cut_date = data[date_col].sort_values().reset_index(drop=True)[int(data.shape[0] *.80)]

train_inds = np.where(data[date_col]<=train_cut_date)[0]
test_inds = np.where((data[date_col]>train_cut_date) & (data[date_col]<=test_cut_date))[0]
eval_inds = np.where(data[date_col]>test_cut_date)[0]

features = sparse.csr_matrix(features)
train_x = sparse.coo_matrix(features[train_inds,:])
test_x = sparse.coo_matrix(features[test_inds,:])
eval_x = sparse.coo_matrix(features[eval_inds,:])

gc.collect()
col_ind = list(data.columns).index(target_col)
train = xgb.DMatrix(train_x, feature_names=trans.feature_names,label=data.iloc[train_inds,col_ind].values, weight= weights[train_inds])
test = xgb.DMatrix(test_x, feature_names=trans.feature_names,label=data.iloc[test_inds,col_ind].values, weight= weights[test_inds])
eval = xgb.DMatrix(eval_x, feature_names=trans.feature_names,label=data.iloc[eval_inds,col_ind].values, weight= weights[eval_inds])


#w Build XGboost models
x = XgboostOptimizer()
# update features related to random search algorithm
x.update_params({'objective': 'binary:logistic',
                 'eval_metric':'auc',
                 'max_over_fit': .03,
                 'num_rand_samples' : 40,
                 'eta': np.arange(.04, .25, .01),
                 'max_depth': range(2, 9),
                 'min_child_weight': range(3000, 20000, 300),
                 'num_boost_rounds': 100,
                 'early_stopping_rounds': 2,
                 'max_minutes_total':120})

#call random search algorithm
x.fit_random_search(dtrain = train, evals=[(train, 'Train'), (test, "Test")])


# extract the best model
model = x.best_model
model.feature_names = trans.feature_names


file = 'models/xgb/'+target_col+'_Model.p'
pickle.dump(model, open(file, 'wb'))