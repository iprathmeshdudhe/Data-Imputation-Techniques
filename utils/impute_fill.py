import pandas as pd

def fill_missing_with_imputed_data(path, imputed_data: pd.DataFrame):

    miss_data = pd.read_csv(path)
    imputed_data_cols = imputed_data.columns

    for col in imputed_data_cols:
        miss_data[col] = imputed_data[col]

    return miss_data