import os
import argparse
import pandas as pd

from utils.load_config import get_config, get_yaml
from utils.load_dataset import load_obs_data, load_eval_columns, load_full_data
from utils.create_missing_data import create_missing_dataset
from utils.plot_predictions import plot_line

from imputers.knn import KNN as knn
from imputers.mice import MiceImputer as mice
from imputers.nnimputer import NeuralNetworkImputer
from imputers.simple_imputer import SimpleImpute as s_impute

from evals.eval import Evaluation as eval


import warnings
warnings.filterwarnings("ignore")

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--impute', action='store_true', help='Use the imputation mode.')
    group.add_argument('--eval', action='store_true', help='Evaluate the imputations based in config file.')
    group.add_argument('--create_missing',action='store_true', help='Create Missing data.')


    args = parser.parse_args()

    if args.impute:
        imputer, dataset_name, observation_size, missing_cols, missing_type = get_config("impute")

        if dataset_name == "mimiciv":
            columns = ["vent_etco2", "vent_fio2", "vital_spo2", "vital_hr","vent_rrtot", "blood_paco2", "blood_pao2"]
        else:
            columns = [i for i in range(observation_size)]

        print("Dataset: ", dataset_name.capitalize(), " Imputer: ", imputer)
        dataframe = load_obs_data(dataset_name, columns, missing_type)

        imputers = {
        "ffill" : s_impute.fwd_impute,
        "knn" : knn.impute_data,
        "mice" : mice.mice_impute,
        "miceforest" : mice.mice_forest,
        "nn" : None
        }
        
        if imputer in imputers.keys():
            if imputer == "nn":
                nni = NeuralNetworkImputer(dataset_name, columns, missing_type, missing_cols)

                miss_data_path, train_run, test_run, train_test, model_path = get_yaml()

                missing_data = pd.read_csv(miss_data_path, usecols=columns)
                
                for col in missing_cols:
                    
                    if train_run:
                        nni.fit(col)
                    elif test_run:
                        col_imputations = nni.transform(col, saved_model_path=model_path)
                        missing_data = nni.fill_imputation(missing_data, col, col_imputations)
                    elif train_test:
                        col_imputations = nni.fit_transform(col)
                        missing_data = nni.fill_imputation(missing_data, col, col_imputations)
                
                if train_run:
                    pass
                else:
                    nni.save_imputed_data(missing_data)
                               
            else:
                imputers[imputer](dataframe, dataset_name, missing_type)

    elif args.create_missing:

        dataset_name, dataset_path, observation_size, action_size, missing_cols, missing_type, p_missing, steps = get_config("create_missing")
        print("Dataset: ", dataset_name.capitalize())

        if dataset_name == "mimiciv":
            columns = ["vent_etco2", "vent_fio2", "vital_spo2", "vital_hr","vent_rrtot", "blood_paco2", "blood_pao2"]
        else:
            columns = [i for i in range(observation_size)]

        dataframe = load_full_data(dataset_path, columns)

        create_missing_dataset(dataset_name, dataframe, observation_size, missing_cols, steps=None, random=True, percent = p_missing)

    elif args.eval:
        dataset_name, observation_size, missing_cols, imputer, imputed_data_path = get_config("eval")
        #og_col_1, og_col_2, imputed_col_1, imputed_col_2 = load_eval_columns(dataset_name, observation_size, imputed_data_path)
        eval_cols_dict = load_eval_columns(dataset_name, missing_cols, imputed_data_path)

        
        for i, col in enumerate(missing_cols, start=1):
            og_col = eval_cols_dict[f"og_{col}_{i}"]
            imputed_col = eval_cols_dict[f"imputed_{col}_{i}"]
            eval.scores(og_col, imputed_col, imputed_data_path, col)
            plot_line(imputed_col[:200], og_col[:200], dataset_name, imputer, col)
