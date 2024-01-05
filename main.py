import pandas as pd
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.get_stats import write_missing_data_stats
from utils.load_config import get_config
from utils.load_dataset import load_obs_data, load_eval_columns
from utils.create_missing_data import create_missing_dataset
from imputers.knn import KNN as knn
from imputers.mice import MiceImputer as mice
from imputers.nnimputer import NeuralNetworkImputer

from evals.eval import Evaluation as eval


import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--impute', action='store_true', help='Use the imputation mode.')
    #group.add_argument('--knn_impute',action='store_true', help='Use the KNN imputer.')
    #group.add_argument('--mice_impute',action='store_true', help='Use the MICE imputer.')
    #group.add_argument('--mf_impute',action='store_true', help='Use the MICE Forest imputer.')
    #group.add_argument('--nn_impute',action='store_true', help='Use the Neural Network imputer.')
    group.add_argument('--eval', action='store_true', help='Evaluate the imputations based in config file.')
    group.add_argument('--create_missing',action='store_true', help='Create Missing data.')


    args = parser.parse_args()

    if args.impute:
        imputer, dataset_name, p_missing, observation_size, missing_type = get_config("impute")

        print("Dataset: ", dataset_name.capitalize(), " Imputer: ", imputer)
        dataframe = load_obs_data(dataset_name, observation_size, missing_type)

        imputers = {
        "knn" : knn.impute_data,
        "mice" : mice.mice_impute,
        "miceforest" : mice.mice_forest,
        "nni" : None
        }
        
        if imputer in imputers.keys():
            if imputer == "nni":
                nni = NeuralNetworkImputer(dataset_name, observation_size, missing_type)
                nni.fit(1)
                col1_imputations = nni.transform(1)
                #nni.compare(col1_imputations)
                nni.fit(2)
                col2_imputations = nni.transform(2)
                nni.save_imputation(col1_imputations, col2_imputations)

                               
            else:
                imputers[imputer](dataframe, dataset_name, missing_type)

    elif args.create_missing:

        dataset_name, dataset_path, observation_size, action_size, missing_type, p_missing, steps = get_config("create_missing")
        print("Dataset: ", dataset_name.capitalize())
        dataframe = load_obs_data(dataset_name, observation_size)

        create_missing_dataset(dataset_name, dataset_path, observation_size, steps=None, random=True, percent = 60)

    elif args.eval:
        dataset_name, observation_size, imputed_data_path = get_config("eval")
        og_col_1, og_col_2, imputed_col_1, imputed_col_2 = load_eval_columns(dataset_name, observation_size, imputed_data_path)
        eval.scores(og_col_1, imputed_col_1, imputed_data_path)
        eval.scores(og_col_2, imputed_col_2, imputed_data_path)



    

    

    '''if args.knn_impute:
        knn.impute_data(dataframe, dataset_name)
    elif args.mice_impute:
        mice.mice_impute(dataframe, dataset_name)
    elif args.mf_impute:
        mice.mice_forest(dataframe, dataset_name)
    elif args.nn_impute:
        nni = NeuralNetworkImputer(dataset_name, observation_size)
        nni.fit()
        nni.predict()
    elif args.eval:
        og_col_1, og_col_2, imputed_col_1, imputed_col_2 = load_eval_columns(dataset_name, observation_size, args.eval)
        eval.scores(og_col_1, imputed_col_1, args.eval, dataset_name)
        eval.scores(og_col_2, imputed_col_2, args.eval, dataset_name)
    elif args.create_missing:
        create_missing_dataset(dataset_name, dataset_path, observation_size, steps=None, random=True, percent = 60)'''
    
    '''df = pd.read_csv(f"dataset/{dataset_name}_w_missing_values_random.csv")
    print(write_missing_data_stats(df, dataset_name))'''


