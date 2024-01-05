import pandas as pd
import numpy as np
import warnings
from typing import Optional

warnings.filterwarnings("ignore")



def create_missing_values_random(df, col_index, percentage):
    """
    Create missing values in the specified column of a DataFrame.

    Parameters:
    - df: DataFrame
        The input DataFrame.
    - col_index: int
        The index of the column where missing values should be created.
    - percentage: int
        The percentage of missing values to be generated.

    Returns:
    - df: DataFrame
        The modified DataFrame with missing values.
    - missing_indexes: list
        List of indexes where missing values were created.

    Example usage:
    Assuming you have a DataFrame df, and you want to create 10% missing values in column 2
    df, missing_indexes = create_missing_values(df, 2, 10)
    """

    # Validate column index
    if col_index < 0 or col_index >= len(df.columns):
        raise ValueError("Invalid column index")

    # Calculate the number of missing values to create
    num_rows = df.shape[0]
    num_missing = int((percentage / 100) * num_rows)

    # Get random indices to insert missing values
    missing_indexes = np.random.choice(num_rows, num_missing, replace=False)
    #print(missing_indexes)

    # Create missing values in the specified column
    df.iloc[missing_indexes, col_index] = np.nan

    return df

def create_missing_values_steps(df_column: pd.Series, start_row: int, steps: int):
    """
    Inserts missing values once in every 5 steps starting from a random row in a Pandas DataFrame column.

    Parameters:
    - df_column (pd.Series): The Pandas DataFrame column for which missing values will be inserted.

    Returns:
    - pd.Series: A new column with missing values inserted at specified intervals.
    
    Example Usage:
    ```python
    import pandas as pd

    # Assuming df is your DataFrame and 'column_name' is the name of the column
    df['column_name'] = create_missing_cells(df['column_name'])
    """
    # Get the number of rows in the column
    num_rows = len(df_column)

    # Create a list to store the indices where missing values will be inserted
    # Insert missing values once in every 5 steps starting from the random row
    missing_indices = [i for i in range(start_row, num_rows, steps)]

    # Update the column with missing values
    df_column.iloc[missing_indices] = np.nan

    return df_column

def create_missing_dataset(dataset: str, dataset_path: str, num_of_cols: int, steps: Optional[int], random: bool, percent: Optional[int]):

    # Taking the middle indexed column to modify
    col_index = num_of_cols // 2

    dataframe = pd.read_csv(dataset_path)

    col_to_modify_1 = dataframe.iloc[:, col_index]
    col_to_modify_2 = dataframe.iloc[:, col_index + 1]

    if steps:
        col_to_modify_1 = create_missing_values_steps(col_to_modify_1, 0, 5)
        col_to_modify_2 = create_missing_values_steps(col_to_modify_2, 2, 5)

        dataframe.iloc[:, col_index] = col_to_modify_1
        dataframe.iloc[:, col_index + 1] = col_to_modify_2

        dataframe.to_csv(f"dataset/{dataset}_w_missing_values_step.csv", index=False)

    elif random:
        dataframe = create_missing_values_random(dataframe, col_index, percent)
        dataframe = create_missing_values_random(dataframe, col_index+1, percent)

        dataframe.to_csv(f"dataset/{dataset}_w_missing_values_random.csv", index=False)

    


    #dataframe.to_csv(f"dataset/{dataset}_w_missing_values.csv", index=False)
    print("Missing Data saved in \"Dataset\" Folder")