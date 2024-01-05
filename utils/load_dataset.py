import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_obs_act_data(dataset: str,  feature_size: int , action_size: int):
    # Create a list of column indices based on feature_size and action_size
    columns = [i for i in range(feature_size + action_size)]

    # Construct the file path
    file_path = f"dataset/{dataset}_w_missing_values_random.csv"

    # Read the CSV file using pd.read_csv with specified columns
    df = pd.read_csv(file_path, usecols=columns)

    return df

def load_obs_data(dataset: str,  feature_size: int, missing_type: str):
    # Create a list of column indices based on feature_size
    columns = [i for i in range(feature_size)]

    # Construct the file path
    file_path = f"dataset/{dataset}_w_missing_values_{missing_type}.csv"

    # Read the CSV file using pd.read_csv with specified columns
    df = pd.read_csv(file_path, usecols=columns)

    return df

def load_non_missing_data(dataset: str, feature_size: int, splitted: bool, predict_data: bool, target_col_index: int = None):

    # Taking the middle indexed column to modify
    col_index = feature_size // 2
    columns = [i for i in range(feature_size)]

    file_path = f"dataset/{dataset}_w_missing_values_random.csv"
    df = pd.read_csv(file_path, usecols=columns)
    #print(df.shape)
    df = df[df.iloc[:, col_index].notna() & df.iloc[:, col_index+1].notna()]

    if target_col_index is None or target_col_index not in [1, 2]:
        target_col_index = col_index  # Default to col_index if invalid value provided
    
    target_col_index = col_index if target_col_index == 1 else col_index + 1

    if splitted and not predict_data:
        X = df.drop(df.columns[[col_index, col_index+1]], axis=1)
        y = df.iloc[:, target_col_index]

        #X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.0, random_state=42, shuffle=True)
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)
        return X_tr, X_val, y_tr, y_val
    
    elif predict_data and not splitted:
        dataframe = pd.read_csv(file_path, usecols=columns)
        dataframe = dataframe[dataframe.iloc[:, target_col_index].isna()]

        test_data = dataframe.drop(dataframe.columns[[col_index, col_index+1]], axis=1)
        return test_data
    
    elif splitted and predict_data:
        print("Error: Both splitted and predict_data cannot be true at same time.")
    
    else:
        return df
    
def load_missing_data(dataset: str, feature_size: int):

    # Taking the middle indexed column to modify
    col_index = feature_size // 2
    columns = [i for i in range(feature_size)]

    missing_data = pd.read_csv(f"dataset/{dataset}_w_missing_values_random.csv", usecols=[col_index , col_index+1])
    df = pd.read_csv(f"dataset/{dataset}_w_missing_values_random.csv", usecols=columns)

    # Get row index of np.nan values for the missing values columns
    miss_col1_row_index, miss_col2_row_index = missing_data.iloc[:, 0].isna(), missing_data.iloc[:, 1].isna()
    df_col1_miss = df[miss_col1_row_index]
    df_col2_miss = df[miss_col2_row_index]

    return df_col1_miss, df_col2_miss
    

def load_eval_columns(dataset: str, feature_size: int, imputed_data_path: str):

    # Taking the middle indexed column to modify
    col_index = feature_size // 2
    columns = [i for i in range(feature_size)]

    all_data = pd.read_csv(f"dataset/{dataset}.csv", usecols=[col_index , col_index+1])
    missing_data = pd.read_csv(f"dataset/{dataset}_w_missing_values_random.csv", usecols=[col_index , col_index+1])
    imputed_data = pd.read_csv(imputed_data_path, usecols=[col_index , col_index+1])

    # Get row index of np.nan values for the missing values columns
    miss_col1_row_index, miss_col2_row_index = missing_data.iloc[:, 0].isna(), missing_data.iloc[:, 1].isna()

    og_col_1, og_col_2 = all_data[miss_col1_row_index].iloc[:, 0], all_data[miss_col2_row_index].iloc[:, 1]
    imputed_col_1, imputed_col_2 = imputed_data[miss_col1_row_index].iloc[:, 0], imputed_data[miss_col2_row_index].iloc[:, 1]

    return og_col_1, og_col_2, imputed_col_1, imputed_col_2


