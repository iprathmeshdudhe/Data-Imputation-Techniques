import pandas as pd
import miceforest as mf
from tqdm import tqdm

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from miceforest import mean_match_default, mean_match_shap



class MiceImputer:

    def mice_impute(dataframe: pd.DataFrame, dataset: str, missing_type: str):

        columns = dataframe.columns

        mice =IterativeImputer(random_state=100, max_iter=32, skip_complete=True)
        mice.fit(dataframe)

        X = dataframe.to_numpy()
        n_rows = X.shape[0]
        imputed_data = []

        for i in tqdm(range(n_rows), desc="Imputing", unit="rows"):
            imputed_row = mice.transform([X[i]])
            imputed_data.append(imputed_row[0])
    
        #X_imputated = mice.transform(dataframe)
        imputed_df = pd.DataFrame(imputed_data, columns=columns)

        print("Saving the imputed data \n")
        imputed_df.to_csv(f"Imputation_results/{dataset}_w_imputed_mice_{missing_type}.csv", index=False)
        print(f"Imputed Data saved at location \"Imputation_results/{dataset}_w_imputed_mice_{missing_type}.csv\"")


    def mice_forest(dataframe: pd.DataFrame, dataset: str,  missing_type: str):
        print("mf")
        mean_match_custom = mean_match_default.copy()
        mean_match_custom.set_mean_match_candidates(32)

        columns = dataframe.columns

        kds = mf.ImputationKernel(
            dataframe,
            save_all_iterations=True,
            random_state=100,
            
        )

        kds.mice(32)

        X_imputated = kds.complete_data()
        imputed_df = pd.DataFrame(X_imputated, columns=columns)
        print("Saving the imputed data \n")
        imputed_df.to_csv(f"Imputation_results/{dataset}_w_imputed_mf_{missing_type}.csv", index=False)
        print(f"Imputed Data saved at location \"Imputation_results/{dataset}_w_imputed_mf_{missing_type}.csv\"")
