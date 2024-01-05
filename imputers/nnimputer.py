import pandas as pd
import numpy as np
import os
from datetime import datetime
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, GRU, InputLayer
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError, R2Score
from keras.optimizers import Adam
from utils.plot_predictions import plot_line

from utils.load_dataset import load_non_missing_data, load_eval_columns

import warnings
warnings.filterwarnings("ignore")


class NeuralNetworkImputer:

    def __init__(self, dataset: str, feature_size: int, missing_type: str) -> None:
        self.data_name = dataset
        self.time = datetime.now().strftime("%d%m%y_%H%M%S")
        self.m_type = missing_type
        self.obs_size = feature_size
        
        '''self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = load_non_missing_data(dataset, feature_size, splitted=True, target_col_index=target_col)
        #print(self.X_train.shape, self.X_val.shape, self.X_test.shape, self.y_train.shape, self.y_val.shape, self.y_test.shape)
        
        self.X_train = self.X_train.values.reshape((self.X_train.shape[0], 1, self.X_train.shape[1])) 
        self.X_val = self.X_val.values.reshape((self.X_val.shape[0], 1, self.X_val.shape[1]))
        self.X_test  = self.X_test.values.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))
        #print(self.X_train.shape, self.X_val.shape, self.X_test.shape, self.y_train.shape, self.y_val.shape, self.y_test.shape)
        
        self.model = Sequential()
        self.model.add(InputLayer((1, self.X_train.shape[2])))
        self.model.add(GRU(64))
        self.model.add(Dense(8, 'relu'))
        self.model.add(Dense(1, 'linear'))
        print("Neural Netword based Imputation. \n")'''

    def fit(self, target_col: int = None):

        #self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = load_non_missing_data(self.data_name, self.obs_size, splitted=True, predict_data=False, target_col_index=target_col)
        self.X_train, self.X_val, self.y_train, self.y_val = load_non_missing_data(self.data_name, self.obs_size, splitted=True, predict_data=False, target_col_index=target_col)
        #print(self.X_train.shape, self.X_val.shape, self.X_test.shape, self.y_train.shape, self.y_val.shape, self.y_test.shape)
        
        self.X_train = self.X_train.values.reshape((self.X_train.shape[0], 1, self.X_train.shape[1])) 
        self.X_val = self.X_val.values.reshape((self.X_val.shape[0], 1, self.X_val.shape[1]))
        #self.X_test  = self.X_test.values.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))
        #print(self.X_train.shape, self.X_val.shape, self.X_test.shape, self.y_train.shape, self.y_val.shape, self.y_test.shape)
        
        self.model = Sequential()
        self.model.add(InputLayer((1, self.X_train.shape[2])))
        self.model.add(GRU(64))
        self.model.add(Dense(8, 'relu'))
        self.model.add(Dense(1, 'linear'))
        print("Neural Netword based Imputation. \n")

        cp = ModelCheckpoint("nnModels/", save_best_only=True)
        self.model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError(), R2Score()])
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs=10, callbacks=[cp])

        directory_path = f"nnModels/{self.data_name}_{self.m_type}_model{self.time}/"
        os.makedirs(directory_path, exist_ok=True)
        self.model.save(f"{directory_path}col{target_col}_model.keras")

    def predict(self):
        #imputations = self.model.predict(self.X_test).flatten()
        #self.test_results = pd.DataFrame(data={'Test Predictions':imputations, 'Actuals':self.y_test})
        pass

    def transform(self, target_col: int = None):
        test_data = load_non_missing_data(self.data_name, self.obs_size, False, True, target_col)
        test_data = test_data.values.reshape(test_data.shape[0], 1, test_data.shape[1])

        predicted_data = self.model.predict(test_data).flatten()

        return predicted_data

    def compare(self, imp):
        og_col_1, og_col_2, imputed_col_1, imputed_col_2 = load_eval_columns(self.data_name, self.obs_size, "Imputation_results/Hopper-v4_w_imputed_nn_random.csv")

        plot_line(imp[:50], og_col_1[:50])
        

    def save_imputation(self, imputed_col1, imputed_col2):

        # Taking the middle indexed column to modify
        col_index = self.obs_size // 2
        columns = [i for i in range(self.obs_size)]

        missing_data = pd.read_csv(f"dataset/{self.data_name}_w_missing_values_random.csv", usecols=columns)
        # Get row index of np.nan values for the missing values columns
        miss_col1_row_index, miss_col2_row_index = np.where(missing_data.iloc[:, col_index].isna())[0], np.where(missing_data.iloc[:, col_index+1].isna())[0]
        print(miss_col1_row_index.shape, imputed_col1.shape, miss_col2_row_index.shape, imputed_col2.shape)
        # Filling the missing cells with imputed data
        missing_data.iloc[miss_col1_row_index, col_index] = imputed_col1
        missing_data.iloc[miss_col2_row_index, col_index+1] = imputed_col2

        print("Saving the imputed data \n")
        missing_data.to_csv(f"Imputation_results/{self.data_name}_w_imputed_nn_{self.m_type}.csv", index=False)
        print(f"Imputed Data saved at location \"Imputation_results/{self.data_name}_w_imputed_nn_{self.m_type}.csv\"")

