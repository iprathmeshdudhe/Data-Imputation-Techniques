import json
from loguru import logger
import sys

def get_config(mode):
    try:
        config_file = open("config.json")
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
        missing_dtype = create_missing_config["missing_data_type"]
        p_missing = create_missing_config["percent_missing"]
        steps = create_missing_config["steps"]

        return dataset_name, path, obs_size, act_size, missing_dtype, p_missing, steps

    elif mode == "impute":
        imputation_config = config["imputation_config"]
        imputer = imputation_config["imputer"]
        dataset_name = imputation_config["dataset"]
        obs_size = imputation_config["observation_size"]
        missing_dtype = imputation_config["missing_data_type"]
        p_missing = imputation_config["percent_missing"]

        return imputer, dataset_name, p_missing, obs_size, missing_dtype
    
    elif mode == "eval":
        eval_config = config["eval_config"]
        dataset_name = eval_config["dataset"]
        obs_size = eval_config["observation_size"]
        imp_data_path = eval_config["imputed_data_path"]
        return dataset_name, obs_size, imp_data_path
    