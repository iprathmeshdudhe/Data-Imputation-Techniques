# MIMIC - IV Imputation

There are 3 modes to run the main.py
1. create_missing dataset
   - This is mainly used to create a test set for imputation techniques.
   - It requires dataset_name, full_data, columns to add missing values to, missing pattern (step or random), missing percent
2. impute
  - for mimiciv, just directly use the imputer.
  - In main.py, input which imputer do you want to use in the config file.
  - Define the dataset_config which appropriate values.
  - The state_vectors should contain independent x variables and also the missing variables. Here the missing variables are added just to load the data.
3. eval
  - it needs the list of imputers for which you want to check the performance.
  - make sure that the imputers you are passing, you have the imputed data for that technique or imputer.
  - saves results in evals/evaluation.csv
