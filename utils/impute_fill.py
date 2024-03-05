import pandas as pd

def fill_missing_with_imputed_data(dataset_name: str, imputed_data: pd.DataFrame):

    miss_data = pd.read_csv(f"dataset/{dataset_name}_w_missing_values_random.csv")
    imputed_data_cols = imputed_data.columns

    for col in imputed_data_cols:
        miss_data[col] = imputed_data[col]

    return miss_data