
## imports

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score,bidev_corrcoef
from multiprocessing import cpu_count
import pathos.pools as pp
num_cores = cpu_count()
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from prettytable import PrettyTable
pd.set_option('display.expand_frame_repr', False)


# read in data and mgb
data = pd.read_pickle('data/raw_data.p')
mgb = pickle.load(open('Hourly_Predictions_Bundle.p','rb'))

key_col = 'PAT_ENC_CSN_ID'
time_col = 'REPORTING_TIME'
date_col = 'REPORTING_DATE'
week_col = 'REPORTING_WEEK'
date_type = 'Day'
# create day and week columns so that we can aggregate results by them
data[date_col] = pd.to_datetime(data['REPORTING_TIME'].dt.strftime('%Y-%m-%d'))
#map the isocalendar to the date column ('%Y-%m-%d') to get the week of year for each date
data[week_col] = list(map(lambda x: x.isocalendar()[1],data[date_col]))

# target/pred column for the specific simulation
# target_cols = ['MetSIRSdttm_8', 'MetSIRS4_4hr_24','MetSIRS4_4hr_48', 'MetMEWS4_8', 'MetMEWS4_24','MetMEWS4_48']
# pred_cols = ['MetSIRS4_4hr_8_preds', 'MetSIRS4_4hr_24_preds', 'MetSIRS4_4hr_48_preds', 'MetMEWS4_8_preds', 'MetMEWS4_24_preds', 'MetMEWS4_48_preds']
target_cols = ['SIRS_dttm']
pred_cols = ['MetSIRSdttm_preds']
target_ind = 0
target_col = target_cols[0]
pred_col = pred_cols[0]

# the data pulled in is between these two dates( full dataset is to large to run on local machine without partitioning)
sim_start_date = datetime.datetime(2018,5,1)
sim_end_date = datetime.datetime(2018,7,31)

#identify ICU depts... The alerting will not occur in the icu departments, so need to stratify the results by all/non-icu alerts
#['U4PC PEDIATRIC CARDIOVASCULAR ICU','C8PI PEDIATRIC ICU','A4CV CVICU','U4ST SURGICAL TRAUMA ICU','C8NN NEONATAL ICU', 'U6MI MICU', 'A3MS MSICU', 'U8NI NEURO SCIENCE ICU', 'U9NI 9TH FL NEURO ICU']
icu_depts = [dept for dept in data['DepartmentName'].unique() if dept !=None and dept.lower().__contains__('icu')]












### Determining the Optimal Cutoff for Alerts
## this code iterates through various cutoff values and finds the cutoffs that optimize the F1-score and MCC score
# this is technically not a correct utiliization of these measures due to the fact that we are looking at the time
# and pushing the time before alert to be as large as possible, not looking at the balance of sensitivity and specificity.
# departments will need to set this for themselves based on the resources that they have available

### THIS SECTION HAS BEEN COMMENTED OUT IN THE ACTUAL MARKDOWNS

# need to modify the cuttoff values corresponding to the distribution of the model's predictions

# if you get an error, it typically means that you need to modify the cutoff values (either your upper/lower bound is out of bounds)
cutoffs = np.round(np.arange(.05,.26,.02),2)
# optimal_f1_dict  = {}
# optimal_mcc_dict = {}
#
# # use multiprocessing to speed up the calculations
# p = pp.ProcessPool(num_cores)
#
# # dont really need the for-loop here, i set this code up to be dynamic to have multiple target_columns if desired.
# # for our markdowns we typically only have a single target_column per markdown.
# for target_ind in range(len(target_cols)):
#     pred_col = pred_cols[target_ind]
#     target_col = target_cols[target_ind]
#
#     ## both the preds_binary_list and the true_binary_list will get fed in and used to calculate the corresponding f1/mcc score
#     # Both of the following are lists of lists(iterables) for the multiprocessing.map
#     # create a list of 1's and 0's based on the cutoff.(in-line for loop)
#     preds_binary_list = [1*(data[pred_col]>=cutoff).values for cutoff in cutoffs]
#     # create a list of the true target values for comparison.
#     true_binary_list = [data[target_col].values for cutoff in cutoffs]
#
#     # feed the two lists in to each of the functions(mcc/f1),
#     #  the outcome will be a list of the f1/mcc values corresponding to each of the cuttofs
#     f1_scores = p.map(f1_score,true_binary_list,preds_binary_list)
#     mcc_scores = p.map(bidev_corrcoef, true_binary_list, preds_binary_list)
#
#     # extract the optimal cutoff value by finding which cutoff corresponds to the maximum f1/mcc value
#     optimal_f1_dict[target_col]=cutoffs[f1_scores.index(max(f1_scores))]
#     optimal_mcc_dict[target_col] = cutoffs[mcc_scores.index(max(mcc_scores))]
#
# print('Optimal Cutoff using F1 Score: '+str(optimal_f1_dict[target_col]))
# print('Optimal Cutoff using MCC Score: '+str(optimal_mcc_dict[target_col]))
opt_cutoff=.11





## This next bit of logic creates the main table displaying how the model performs on the 'test' dataset.
## the idea here is to
#  1) iterate over various cutoff values and create a preds_binary_flag [1 if pred>=cutoff, 0 o.w.]
#  2) find all unique patients where target_col==1
#  3) for all unique patients in (2), find the time that they first met the actual target(i.e. for MetSIRS4_4hr_24, we want
#     to find the first time that the patient meets MetSIRS4_4hr (assume this is the t0 for their SIRS4 episode)
#  4) Using (1) and (3) determine whether the model would have alerted prior, during/after, or not at all for each of the patients
#  5) place all results into a table.

target_col = target_cols[target_ind]
pred_col = pred_cols[target_ind]


alert_results_table_columns = ['Total Patient Alerts Per '+date_type,'Non-ICU Patient Alerts Per '+date_type,'Proportion Alerts Prior','Proportion Alerts During/After','Proportion Alerts Missed','Average Alert Time Before Target(Hours)','Proportion Patient Wrong Alerts']
# construct the table of dimension (len(cutoffs),len(alert_results_table_columns))
alert_results_table = pd.DataFrame(np.zeros((len(cutoffs),len(alert_results_table_columns))),columns=alert_results_table_columns,index=cutoffs)

# this alert_time_bins  dataframe keeps track of the proportion of times the model fires at certain time intervals prior to the t0.
alert_time_bins = pd.DataFrame(np.zeros((len(cutoffs),6)),columns=['>24 Hours','24-13 Hours','12-9 Hours','8-5 Hours','4-1 Hours','Alerted During/After'],index=cutoffs)

# iterate over all the cutoffs
for cutoff in cutoffs:
    #Construct an empty dictionary that will keep up with the results of the current cutoff
    alerts_dict = {col: 0 for col in alert_results_table_columns}
    # create preds_binary based on whether pred>=cutoff for the current cutoff
    data['preds_binary']= 1*(data[pred_col]>=cutoff).values

    # for all patients, and nonicu patients, find the number of patient alerts per day (ppd)
    # these lists will be a vector containing a count of the number of patients that have target_col==0
    # for each of the days in the dataset
    ppd_list = []
    nonicu_ppd_list = []
    for day in data['REPORTING_DATE'].unique():
        tmp_data = data.iloc[np.where(data['REPORTING_DATE']==day)[0],:].reset_index(drop=True)
        ppd_list.append(len(tmp_data.loc[np.where(tmp_data['preds_binary']==1)[0],'PAT_ENC_CSN_ID'].unique()))
        nonicu_ppd_list.append(len(tmp_data.loc[np.where((tmp_data['preds_binary']==1) & (tmp_data['DepartmentName'].isin(icu_depts)==False))[0],'PAT_ENC_CSN_ID'].unique()))
    # find the average patients per day that would be alerted on. (find the mean of the ppd_list and nonicu_ppd_list)
    alerts_dict['Total Patient Alerts Per '+date_type] = np.mean(ppd_list).round(2)
    alerts_dict['Non-ICU Patient Alerts Per ' + date_type] = np.mean(nonicu_ppd_list).round(2)

    # find the keys of the patients who had the target
    unique_target_keys = data.loc[np.where(data[target_col].isna()==False)[0],key_col].unique()
    # find the keys of the patients that the model alerted on
    unique_alert_keys = data.loc[np.where(data['preds_binary']==1)[0],key_col].unique()
    # iterate over all of the patients that had the target and determine when/if the model alerted on them
    for target_key in unique_target_keys:
        #SIRS_dttm (or time 0 of sepsis episode)
        first_target_time = data.loc[np.where(data[key_col]==target_key)[0],'SIRS_dttm'].unique()[0]

        #The dataset only has times rows for 5-1-2018  - Future, so if you get a SIRSdttm prior to this date just continue
        if datetime.datetime.utcfromtimestamp(first_target_time.tolist()/1e9)<datetime.datetime(2018,5,1):
            continue
        else:
            # find the indices where the model alerted on the individual
            alert_inds = np.where((data[key_col]==target_key) & (data['preds_binary']==1))[0]
            # if no alert_indices, we missed a sepsis patient
            if len(alert_inds)==0:
                alerts_dict['Proportion Alerts Missed'] += 1
            else:
                # calculate the first alert time
                first_alert_time = min(data.loc[alert_inds, time_col])
                # calculate the timedelta between the first alert time and their SIRS_dttm
                alert_diff = (first_alert_time.to_pydatetime() - datetime.datetime.utcfromtimestamp(first_target_time.tolist()/1e9)).seconds / 3600

                # if alert_time<target_time, model alerted prior - find out the amount of hours prior
                if first_alert_time<first_target_time:
                    alerts_dict['Proportion Alerts Prior']+=1
                    if alert_diff<=4:
                        alert_time_bins.loc[cutoff, '4-1 Hours'] += 1
                    elif alert_diff<=8:
                        alert_time_bins.loc[cutoff, '8-5 Hours'] += 1
                    elif alert_diff<=12:
                        alert_time_bins.loc[cutoff, '12-9 Hours'] += 1
                    elif alert_diff <= 24:
                        alert_time_bins.loc[cutoff, '24-13 Hours'] += 1
                    else:
                        alert_time_bins.loc[cutoff, '>24 Hours'] += 1
                # Model alerted during/after SIRS_dttm
                else:
                    alerts_dict['Proportion Alerts During/After'] += 1
                    alert_time_bins.loc[cutoff, 'Alerted During/After'] += 1

                alerts_dict['Average Alert Time Before Target(Hours)'] += alert_diff
    # Calculate the proportion of patients wrongly alerted on by finding the set of keys that were alerted on, but not in the unique_target_keys
    alerts_dict['Proportion Patient Wrong Alerts'] = len(
        (set(unique_alert_keys) - set(unique_alert_keys).intersection(unique_target_keys))) / len(unique_alert_keys)
    alerts_dict['Proportion Patient Wrong Alerts'] = len((set(unique_alert_keys) - set(unique_alert_keys).intersection(unique_target_keys)))/len(unique_alert_keys)
    # These are all raw numbers, so to get the proportion need to divide by the total number of target_keys (patients with Sepsis & SIRSdttm)
    for col in ['Proportion Alerts Missed','Proportion Alerts Prior','Proportion Alerts During/After','Average Alert Time Before Target(Hours)']:
        alert_results_table.loc[cutoff,col] = alerts_dict.get(col)/len(unique_target_keys)
    for col in ['Non-ICU Patient Alerts Per '+date_type,'Total Patient Alerts Per '+date_type,'Proportion Patient Wrong Alerts']:
        alert_results_table.loc[cutoff, col] = alerts_dict.get(col)





# round the results to 2 decimal places and apply a lightgreen background to the cutoff that is chosen for the optimal cutoff

alert_results_table = alert_results_table.round(2)
alert_results_table.style.apply(lambda x: ['background: lightgreen' if x.name == opt_cutoff else '' for i in x],
               axis=1)



### Breakdown of Alert Times for SIRS4_4hr Patients
alert_time_bins = alert_time_bins/len(unique_target_keys)
alert_time_bins = alert_time_bins.round(2)
alert_time_bins.style.apply(lambda x: ['background: lightgreen' if x.name == opt_cutoff else '' for i in x],
               axis=1)




### Model Performance on Patients with Sepsis Primary DRG

# This next bit of code performs a similar function to the code above that iterates over the cutoff values and finds
# the number of patients that would have been alerted on prior, during/after, or not at all. This time, the code is
# specifically targeting the sepsis population and finding proportion of times that the model would have alerted with
# respect to the SIRS_dttm for the sepsis patients.




sepsis_data = pd.read_pickle('data/sepsis_patient_data.p')
sepsis_keys = sepsis_data[key_col].unique()
mort_sepsis_keys = sepsis_data.loc[np.where(sepsis_data['Death_Num']==1)[0],key_col].unique()
sirs4_4hr_count = 0
# this can be omitted, but this just finds the number of sepsis patients that meet the SIRS4_4hr criteria during their stay
for key in sepsis_keys:
    sirs4_4hr_count+= 1 if sepsis_data.loc[np.where(sepsis_data[key_col]==key)[0],'MetSIRS4_4hr'].sum()>0 else 0




target_col = target_cols[target_ind]
pred_col = pred_cols[target_ind]

# initialize the results table for the sepsis population
alert_results_table_columns = ['Total Sepsis Primary DRGs','Number Sepsis Patients Alerted On','Total Deceased Sepsis Primary DRGs','Number Deceased Sepsis Patients Alerted On','Proportion Alerts Prior','Proportion Alerts During/After','Proportion Alerts Missed','Average Alert Time Before Target(Hours)','Proportion Sepsis Patients Missed']
alert_results_table = pd.DataFrame(np.zeros((len(cutoffs),len(alert_results_table_columns))),columns=alert_results_table_columns,index=cutoffs)
# create the alert time_bins
alert_time_bins = pd.DataFrame(np.zeros((len(cutoffs),6)),columns=['>24 Hours','24-13 Hours','12-9 Hours','8-5 Hours','4-1 Hours','Alerted After'],index=cutoffs)

# iterate over the cutoff values and find the information corresponding to each of the values
for cutoff in cutoffs:
    # Create a blank dict, alerts_dict, that keeps track of the temporary results within each of the iterations
    alerts_dict = {col: 0 for col in alert_results_table_columns}
    # find the alerts_binary - 1 if pred>=cutoff else 0
    sepsis_data['preds_binary']= 1*(sepsis_data[pred_col]>=cutoff).values

    ## find some descriptive statistics on the sepsis population ( self explained by the column name what is going on)
    alerts_dict['Total Sepsis Primary DRGs'] = len(sepsis_keys)
    alerts_dict['Number Sepsis Patients Alerted On'] = len(sepsis_data.loc[np.where(sepsis_data['preds_binary']==1)[0],key_col].unique())
    alerts_dict['Proportion Sepsis Patients Missed'] = (alerts_dict['Total Sepsis Primary DRGs'] - alerts_dict['Number Sepsis Patients Alerted On'])/alerts_dict['Total Sepsis Primary DRGs']
    alerts_dict['Total Deceased Sepsis Primary DRGs'] = len(mort_sepsis_keys)
    alerts_dict['Number Deceased Sepsis Patients Alerted On'] = len(sepsis_data.loc[np.where((sepsis_data['preds_binary'] == 1) & (sepsis_data['Death_Num']==1))[0], key_col].unique())

    # find unique target keys ( that have a SIRSdttm)
    unique_target_keys = sepsis_data.loc[np.where(sepsis_data['SIRS_dttm'].isnull()==False)[0],key_col].unique()
    # find the unique set of patients that were alerted on using the cutoff established
    unique_alert_keys = sepsis_data.loc[np.where(sepsis_data['preds_binary']==1)[0],key_col].unique()

    # iterate over all of the patient_keys that have SIRS_dttm to calculate whether the model alerted prior,during/after, or not at all for each patient
    for target_key in unique_target_keys:
        # find the patient's time 0 (SIRS_dttm)
        first_target_time = sepsis_data.loc[np.where(sepsis_data[key_col] == target_key)[0], 'SIRS_dttm'].unique()[0]
        # find the indices where the model alerted on this patient ( using whethere the preds_binary==1)
        alert_inds = np.where((sepsis_data[key_col]==target_key) & (sepsis_data['preds_binary']==1))[0]
        #if no alerts, we missed the patient
        if len(alert_inds)==0:
            alerts_dict['Proportion Alerts Missed'] += 1
        # we alerted on the patient, now need to find whether it was prior, or after SIRS_dttm
        else:
            # find first alert time from model
            first_alert_time = min(sepsis_data.loc[alert_inds, time_col])
            # calculate the timedelta between alert and SIRS_dttm
            alert_diff = (datetime.datetime.utcfromtimestamp(first_target_time.tolist() / 1e9) - first_alert_time.to_pydatetime()).total_seconds() / 3600
            # IF, model alerted prior for this patient, further calculate what time bin it should be placed in
            if first_alert_time<first_target_time:
                alerts_dict['Proportion Alerts Prior']+=1
                if alert_diff<=4:
                    alert_time_bins.loc[cutoff, '4-1 Hours'] += 1
                elif alert_diff<=8:
                    alert_time_bins.loc[cutoff, '8-5 Hours'] += 1
                elif alert_diff<=12:
                    alert_time_bins.loc[cutoff, '12-9 Hours'] += 1
                elif alert_diff <= 24:
                    alert_time_bins.loc[cutoff, '24-13 Hours'] += 1
                else:
                    alert_time_bins.loc[cutoff, '>24 Hours'] += 1
            # ELSE, alerted after
            else:
                alerts_dict['Proportion Alerts During/After'] += 1
                alert_time_bins.loc[cutoff, 'Alerted After'] += 1

            alerts_dict['Average Alert Time Before Target(Hours)'] += alert_diff

    # these are all raw numbers, to calculate the proportion need to divide by the total number of sepsis patients
    for col in ['Proportion Alerts Missed','Proportion Alerts Prior','Proportion Alerts During/After','Average Alert Time Before Target(Hours)']:
        alert_results_table.loc[cutoff,col] = alerts_dict.get(col)/len(unique_target_keys)
    for col in ['Total Sepsis Primary DRGs','Number Sepsis Patients Alerted On','Total Deceased Sepsis Primary DRGs','Number Deceased Sepsis Patients Alerted On','Proportion Sepsis Patients Missed']:
        alert_results_table.loc[cutoff, col] = alerts_dict.get(col)

alert_results_table = alert_results_table.round(2)
alert_results_table.style.apply(lambda x: ['background: lightgreen' if x.name == opt_cutoff else '' for i in x],
               axis=1)


### Breakdown of Alert Times for SIRS4_4hr Patients
alert_time_bins = alert_time_bins/len(unique_target_keys)
alert_time_bins = alert_time_bins.round(2)
alert_time_bins.style.apply(lambda x: ['background: lightgreen' if x.name == opt_cutoff else '' for i in x],
               axis=1)
