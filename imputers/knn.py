import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from tqdm import tqdm
#import matplotlib.pyplot as plt
#from cuml.impute import KNNImputer as cuKNNImputer


class KNN:

    def get_score(X, Y):
    
        # evaluate each strategy on the dataset
        results = list()
        strategies = [str(i) for i in [1,3,5,7,9,15,18,21]]
        for s in strategies:
            # create the modeling pipeline
            pipeline = Pipeline(steps=[('i', KNNImputer(n_neighbors=int(s))), ('m', RandomForestClassifier())])
            # evaluate the model
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
            scores = cross_val_score(pipeline, X, Y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
            # Use cross_val_predict to get imputed values for each fold
            # store results
            results.append(scores)
            print('>%s %.3f (%.3f)' % (s, np.mean(scores), np.std(scores)))

    def transform_with_progress_bar(imputer, X):
        n_rows = X.shape[0]
        imputed_data = []
        for i in tqdm(range(n_rows), desc="Imputing", unit="rows"):
            imputed_row = imputer.transform([X[i]])
            imputed_data.append(imputed_row[0])
        return imputed_data



    def impute_data(df: pd.DataFrame, dataset: str, missing_type: str):
        
        imputer = KNNImputer()
        imputer.fit(df)
        #X_imputed = imputer.transform(df.head(50000))
        # Call the custom transform function with a progress bar
        X_imputed = KNN.transform_with_progress_bar(imputer, df.to_numpy())

        imputed_df = pd.DataFrame(X_imputed, columns=df.columns)
        print("Saving the imputed data \n")
        imputed_df.to_csv(f"Imputation_results/{dataset}_w_imputed_knn_{missing_type}.csv", index=False)
        print(f"Imputed Data saved at location \"Imputation_results/{dataset}_w_imputed_knn_{missing_type}.csv\"")

    

