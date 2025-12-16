--Copied over from clarity server to muscedw by Wei Ding 10/1/2024
SELECT DISTINCT
s.pat_id,
s.pat_mrn_id,
s.dx_name,
lp.PAT_ENC_CSN_ID,
lv.REPORTING_TIME,
s.SIRS_dttm,
s.Adm_To_ICU,
s.ICU_time,
s.Death_Num,
s.ED_DEPARTURE_TIME,
s.Blood_Cult_Drawn_dttm,
s.Antibiotic_admin_dttm,
s.Bolus_admin_dttm,
s.Lactate_Result_dttm,
s.vaso_admin_dttm,
s.SIRS_Department_Name,
s.Minutes_SIRS_Antibiotic,
s.Minutes_SIRS_Blood_Cult,
s.Minutes_SIRS_Bolus,
s.Minutes_SIRS_Lactate,
s.Minutes_SIRS_vaso,
lp.Prediction,
lp.PredictionPercentile,
lp.ModelName,
lp.TopFeature1,
lp.TopFeature2,
lp.TopFeature3,
lp.TopFeature4,
lp.TopFeature5,
lp.TopFeature6,
lp.TopFeature7,
lp.TopFeature8,
lp.TopFeature9,
lp.TopFeature10,
lv.[MetSIRS4_4hr_24],
lv.[MetSIRS4_4hr],
lv.[MetSIRS3_4hr_24],
lv.[MetSIRS3_4hr],
lv.[BLOOD PRESSURE (SYSTOLIC)],
lv.[BLOOD PRESSURE (DIASTOLIC)],
 lv.[PO2 (CORR), ARTERIAL],
 lv.MetSIRS_WBC,
 lv.MaxRR8,
 lv.MinTemp8,
 lv.MinHR48,
 lv.TEMPERATURE,
 lv.MinHR24,
 lv.[CPM S16 R INV O2 DEVICE],
 lv.MEWS_Temp,
 lv.MEWS_RR,
 lv.MetSIRS_Temp,
 lv.MinTemp24,
 lv.[CPM S16 R AS SC RASS (RICHMOND AGITATION-SEDATION SCALE) (TRANSFORMED)],
 lv.[CPM S16 R AS SC BRADEN SCORE],
 lv.MaxHR8,
 lv.[CPM S16 R AS SC GLASGOW COMA SCALE SCORE],
 lv.MinHR8,
 lv.RESPIRATIONS,
 lv.MaxHR24,
 lv.MinTemp48,
 lv.PULSE,
 lv.MaxTemp8
FROM
	StatisticalModels.dbo.SepsisOutcomes_Allv1 s
INNER JOIN
	StatisticalModels.dbo.LabsVitalsHourly_Predictions lp
ON
	lp.PAT_ENC_CSN_ID = s.PAT_ENC_CSN_ID
INNER JOIN
	StatisticalModels.dbo.vwLabsVitalsHourly_Adult lv
ON
	lv.PAT_ENC_CSN_ID = lp.PAT_ENC_CSN_ID
AND
	lv.REPORTING_TIME = lp.REPORTING_TIME

INNER JOIN
    [vs-claritydb\clarity].Clarity.dbo.V_PAT_ADT_LOCATION_HX ADT
ON
    lp.PAT_ENC_CSN_ID = ADT.PAT_ENC_CSN
AND
    lv.REPORTING_TIME between ADT.IN_DTTM AND ADT.OUT_DTTM

WHERE
	lp.ModelName = 'MetSIRS4_4hr_24'
