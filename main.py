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
from imputers.ssl_imputer import SSLImputer
from imputers.simple_imputer import SimpleImpute as s_impute

from evals.eval import Evaluation as eval
from dataset_config import MimicConfig, TudConfig, HopperConfig


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
        imputer, dataset_name, _, _, _ = get_config("impute")

        if dataset_name == "mimiciv":
            dataset_cfg = MimicConfig()
            #columns = ["vent_etco2", "vent_fio2", "vital_spo2", "vital_hr","vent_rrtot", "blood_paco2", "blood_pao2"]
        elif dataset_name == "tud":
            dataset_cfg = TudConfig()
            #columns = [i for i in range(observation_size)]
        elif dataset_name == "hopper":
            dataset_cfg = HopperConfig()

        columns = dataset_cfg.state_vector
        print("Dataset: ", dataset_name.capitalize(), " Imputer: ", imputer)
        dataframe = load_obs_data(dataset_cfg)

        imputers = {
        "ffill" : s_impute.fwd_impute,
        "knn" : knn.impute_data,
        "mice" : mice.mice_impute,
        "miceforest" : mice.mice_forest,
        "nn" : None,
        "ssl" : None
        }
        
        if imputer in imputers.keys():
            if imputer == "nn":
                nni = NeuralNetworkImputer(dataset_cfg)

                train_run, test_run, train_test, model_path = get_yaml()

                missing_data = pd.read_csv(dataset_cfg.missing_data_path, usecols=dataset_cfg.state_vector)
                
                for col in dataset_cfg.missing_state_vector:
                    
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
                    nni.save_imputed_data(dataset_cfg.imputed_data_path, missing_data)

            elif imputer == "ssl":
                ssl = SSLImputer(dataset_cfg)
                ssl.build_model()
                ssl.fit(dataframe)

                # TODO: Add Intermediate step to fill the missing value for prediction
                ssl.transform(dataframe)

                               
            else:
                imputers[imputer](dataframe, dataset_cfg)

    elif args.create_missing:

        dataset_name, _, _, _, _, missing_type, p_missing, steps = get_config("create_missing")
        print("Dataset: ", dataset_name.capitalize())

        if dataset_name == "mimiciv":
            dataset_cfg = MimicConfig()
        elif dataset_name == "tud":
            dataset_cfg = TudConfig()
        elif dataset_name == "hopper":
            dataset_cfg = HopperConfig()
            
        columns = dataset_cfg.state_vector

        dataframe = load_full_data(dataset_cfg.full_data_path, columns)

        create_missing_dataset(dataset_cfg.data_name, dataframe, dataset_cfg.missing_state_vector, steps=None, random=True, percent = p_missing)

    elif args.eval:
        dataset_name, _, _, imputer, _ = get_config("eval")
        #og_col_1, og_col_2, imputed_col_1, imputed_col_2 = load_eval_columns(dataset_name, observation_size, imputed_data_path)
        

        if dataset_name == "mimiciv":
            dataset_cfg = MimicConfig()
        elif dataset_name == "tud":
            dataset_cfg = TudConfig()
        elif dataset_name == "hopper":
            dataset_cfg = HopperConfig()

        eval_cols_dict = load_eval_columns(dataset_cfg, imputer)
        for i, col in enumerate(dataset_cfg.missing_state_vector, start=1):
            og_col = eval_cols_dict[f"og_{col}_{i}"]
            imputed_col = eval_cols_dict[f"imputed_{col}_{i}"]
            eval.scores(og_col, imputed_col, dataset_cfg.imputed_data_path, imputer, col)
            plot_line(imputed_col[:200], og_col[:200], dataset_name, imputer, col)