--written by Austin Mishoe, 9/27/2018
Select  lv.PAT_ENC_CSN_ID,
		lv.REPORTING_TIME,
		MetSIRS4_4hr_8_preds,
		MetSIRS4_4hr_24_preds,
		MetSIRS4_4hr_48_preds,
		MetMEWS4_8_preds,
		MetMEWS4_24_preds,
		MetMEWS4_48_preds,
		[MetSIRS4_4hr],
		[SIRSScore],
		[MetMEWS4],
		[MEWSScore],
		[MetSIRS4_4hr_8],
		[MetSIRS4_4hr_24],
		[MetSIRS4_4hr_48],
		[MetSIRS3_4hr_8],
		[MetSIRS3_4hr_24],
		[MetSIRS3_4hr_48],
		[NoSIRS4_12],
		[MetMEWS4_8],
		[MetMEWS4_24],
		[MetMEWS4_48],
		ADT.ADT_DEPARTMENT_NAME as DepartmentName
FROM

(SELECT
	  [PAT_ENC_CSN_ID]
	  ,[REPORTING_TIME]
      ,[MetSIRS4_4hr]
      ,[SIRSScore]
      ,[MetMEWS4]
      ,[MEWSScore]
      ,[MetSIRS4_4hr_8]
      ,[MetSIRS4_4hr_24]
      ,[MetSIRS4_4hr_48]
      ,[MetSIRS3_4hr_8]
      ,[MetSIRS3_4hr_24]
      ,[MetSIRS3_4hr_48]
      ,[NoSIRS4_12]
      ,[MetMEWS4_8]
      ,[MetMEWS4_24]
      ,[MetMEWS4_48]
	  FROM
		StatisticalModels.dbo.vwLabsVitalsHourly_Adult
) lv

left join
(SELECT
		PAT_ENC_CSN_ID,
		REPORTING_TIME,
		sum(MetSIRS4_4hr_8) AS MetSIRS4_4hr_8_preds,
		sum(MetSIRS4_4hr_24) AS MetSIRS4_4hr_24_preds,
		sum(MetSIRS4_4hr_48) AS MetSIRS4_4hr_48_preds,
		sum(MetMEWS4_8) AS MetMEWS4_8_preds,
		sum(MetMEWS4_24) AS MetMEWS4_24_preds,
		sum(MetMEWS4_48) AS MetMEWS4_48_preds
FROM   StatisticalModels.dbo.LabsVitalsHourly_Predictions
PIVOT
(
       SUM(PredictionPercentile)
       FOR ModelName IN (MetSIRS4_4hr_8, MetSIRS4_4hr_24, MetSIRS4_4hr_48,MetMEWS4_8, MetMEWS4_24,MetMEWS4_48)
) AS P
group by PAT_ENC_CSN_ID,REPORTING_TIME
) lv_preds

ON
	lv_preds.PAT_ENC_CSN_ID=lv.PAT_ENC_CSN_ID and
	lv_preds.REPORTING_TIME = lv.REPORTING_TIME

	left join
    [vs-claritydb\clarity].Clarity.dbo.V_PAT_ADT_LOCATION_HX ADT
        on lv.PAT_ENC_CSN_ID = ADT.PAT_ENC_CSN
        and lv.REPORTING_TIME between ADT.IN_DTTM AND ADT.OUT_DTTM

WHERE lv.REPORTING_TIME BETWEEN '2018-05-01' AND '2018-07-31'
order by lv.REPORTING_TIME