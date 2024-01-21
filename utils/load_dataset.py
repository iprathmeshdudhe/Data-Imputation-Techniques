import pandas as pd
from sklearn.model_selection import train_test_split

def load_obs_act_data(dataset: str,  feature_size: int , action_size: int):
    # Create a list of column indices based on feature_size and action_size
    columns = [i for i in range(feature_size + action_size)]

    # Construct the file path
    file_path = f"dataset/{dataset}_w_missing_values_random.csv"

    # Read the CSV file using pd.read_csv with specified columns
    df = pd.read_csv(file_path, usecols=columns)

    return df

def load_full_data(dataset_path: str,  columns: list):

    # Read the CSV file using pd.read_csv with specified columns
    df = pd.read_csv(dataset_path, usecols=columns)

    return df

def load_obs_data(dataset: str, columns: list, missing_type: str):
    
    # Construct the file path
    file_path = f"dataset/{dataset}_w_missing_values_{missing_type}.csv"

    # Read the CSV file using pd.read_csv with specified columns
    df = pd.read_csv(file_path, usecols=columns)

    return df

def load_non_missing_data(dataset: str, columns: list, miss_cols: list, splitted: bool, predict_data: bool, target_col: str = None):

    file_path = f"dataset/{dataset}_w_missing_values_random.csv"
    df = pd.read_csv(file_path, usecols=columns)
    #print(df.shape)
    df = df.dropna(subset=miss_cols)

    if splitted and not predict_data:
        X = df.drop(miss_cols, axis=1)
        y = df.loc[:, target_col]

        #X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.0, random_state=42, shuffle=True)
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)
        return X_tr, X_val, y_tr, y_val
    
    elif predict_data and not splitted:
        dataframe = pd.read_csv(file_path, usecols=columns)
        dataframe = dataframe[dataframe.loc[:, target_col].isna()]

        test_data = dataframe.drop(miss_cols, axis=1)
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
    

def load_eval_columns(dataset: str, miss_cols: list, imputed_data_path: str):

    all_data = pd.read_csv(f"dataset/mimic_full_data.csv", usecols=miss_cols)

    if dataset == "mimic":
        all_data = pd.read_csv(f"dataset/mimic_full_data.csv", usecols=miss_cols)
    else:
        all_data = pd.read_csv(f"dataset/{dataset}.csv", usecols=miss_cols)

    missing_data = pd.read_csv(f"dataset/{dataset}_w_missing_values_random.csv", usecols=miss_cols)
    imputed_data = pd.read_csv(imputed_data_path, usecols=miss_cols)

    '''# Get row index of np.nan values for the missing values columns
    miss_col1_row_index, miss_col2_row_index = missing_data.iloc[:, 0].isna(), missing_data.iloc[:, 1].isna()

    og_col_1, og_col_2 = all_data[miss_col1_row_index].iloc[:, 0], all_data[miss_col2_row_index].iloc[:, 1]
    imputed_col_1, imputed_col_2 = imputed_data[miss_col1_row_index].iloc[:, 0], imputed_data[miss_col2_row_index].iloc[:, 1]'''

    eval_col_dict = {}

    i = 1
    for col in miss_cols:
        miss_col_row = missing_data.loc[:, col].isna()
        og_col = all_data[miss_col_row].loc[:, col]
        imputed_col = imputed_data[miss_col_row].loc[:, col]

        eval_col_dict[f"og_{col}_{i}"] = og_col
        eval_col_dict[f"imputed_{col}_{i}"] = imputed_col
        i += 1

    return eval_col_dict


