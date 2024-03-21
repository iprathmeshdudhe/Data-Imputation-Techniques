import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.impute import KNNImputer

from utils.impute_fill import fill_missing_with_imputed_data

class KNN:

    def transform_with_progress_bar(imputer, X):
        n_rows = X.shape[0]
        imputed_data = []
        for i in tqdm(range(n_rows), desc="Imputing", unit="rows"):
            imputed_row = imputer.transform([X[i]])
            imputed_data.append(imputed_row[0])
        return imputed_data



    def impute_data(df: pd.DataFrame, dataset_cfg):

        neighbors = [13]

        for n in neighbors:
            print(f"Performing KNN for {n} Neighbors. \n")
            imputer = KNNImputer(n_neighbors=n)
            imputer.fit(df)
            #X_imputed = imputer.transform(df.head(50000))
            # Call the custom transform function with a progress bar
            X_imputed = KNN.transform_with_progress_bar(imputer, df.to_numpy())

            imputed_df = pd.DataFrame(X_imputed, columns=df.columns)

            filled_data = fill_missing_with_imputed_data(dataset_cfg.missing_data_path, imputed_df)
            print("Saving the imputed data \n")

            filled_data.to_csv(f"{dataset_cfg.imputed_data_path}_knn_{n}.csv", index=False)
            print(f"Imputed Data saved at location \"{dataset_cfg.imputed_data_path}_knn_{n}.csv\"")