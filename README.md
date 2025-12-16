# LabsVitalsHourly_AlertSimulations

This project takes the predictions from the models built in 'Hourly_Predictions' and performs further analysis on performance.
The scripts(markdowns) in this project perform back simulations for each model on historic data to find the proportion of
patients missed, proportion alerted on prior, proportion alerted on during/after, etc... for each given target column.
All of this information in the markdowns is presented to give greater insight on how the models will perform in production.

## All markdowns were generated using using the files in the 'scripts' folder as templates.
1) <b>scripts/SimulationScript_NonDTTM_Targets.py</b> should be used as reference for any new simulation markdowns where the target_col is like SIRS3_24,SIRS4_24,MEWS4_24,etc..
2) <b>scripts/SimulationScript_DTTM_Targets.py</b> should be used as reference for creating any simulations where the target_col is SIRS_dttm, Bolus_dttm, etc..
## How to Modify Existing Markdowns
1) For the HTML file you are looking to modify, open the corresponding .pmd file
2) Make modifications to the pmd file
3) open a command prompt || cd into project_home_directory || type 'pweave -f md2html markdowns/"your_pmd_file.pmd"'

## How to Generate a New Simulation Markdown
Need to get the data necessary, and then create the markdown file.

#### Get necessary data for the simulation
The first step is to make sure that you have the correct data pulled in to run the simulation, as well as the corresponding
predictions. I did this in one of two ways:
1) Write the predictions to a SQL table, join them to the flag and then use this data.
OR (The second is to locally predict and merge the datasets together - What I used when testing many models)
2) Pull in the model || predict on the raw LV dataset || merge predictions into dataframe with target values.
I have written the script 'merge_predictions_with_data.py' that gives a template to do this

#### Generate the markdown file
1) Copy an existing markdown .pmd file and rename.
2) Make modifications to the pmd file
3) open a command prompt || cd into project_home_directory || type 'pweave -f md2html markdowns/"your_pmd_file.pmd"'




## Author

* **Austin Mishoe, Data Science, MUSC**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details



