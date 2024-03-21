import json
import yaml
from loguru import logger
import sys

def get_config(mode):
    try:
        config_file = open("configs/config.json")
        config = json.load(config_file)
    except Exception as exc:
        logger.exception(exc)
        sys.exit(1)

    if mode == "create_missing":
        
        create_missing_config = config["create_missing"]
        dataset_name = create_missing_config["dataset"]
        path = create_missing_config["path"]
        obs_size = create_missing_config["observation_size"]
        act_size = create_missing_config["action_size"]
        missing_cols = create_missing_config["missing_columns"]
        missing_type = create_missing_config["missing_type"]
        p_missing = create_missing_config["percent_missing"]
        steps = create_missing_config["steps"]

        return dataset_name, path, obs_size, act_size, missing_cols, missing_type, p_missing, steps

    elif mode == "impute":
        imputation_config = config["imputation_config"]

        imputer = imputation_config["imputer"]
        dataset_name = imputation_config["dataset"]
        obs_size = imputation_config["observation_size"]
        missing_cols = imputation_config["missing_columns"]
        missing_type = imputation_config["missing_type"]

        return imputer, dataset_name, obs_size, missing_cols, missing_type
    
    elif mode == "eval":
        eval_config = config["eval_config"]
        dataset_name = eval_config["dataset"]
        obs_size = eval_config["observation_size"]
        missing_cols = eval_config["missing_columns"]
        imputer = eval_config["imputer"]
        imp_data_path = eval_config["imputed_data_path"]
        return dataset_name, obs_size, missing_cols, imputer, imp_data_path
    
def get_yaml():

    try:
        config_file = open("configs/nn_config.yml")
        nn_config = yaml.safe_load(config_file)
    except Exception as exc:
        logger.exception(exc)
        sys.exit(1)

    train, test, train_test = nn_config["TRAIN"], nn_config["TEST"], nn_config["TRAIN_TEST"]
    model_path = nn_config["MODEL_PATH"]

    return train, test, train_test, model_path