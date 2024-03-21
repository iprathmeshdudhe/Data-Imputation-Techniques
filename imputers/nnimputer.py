import pickle
import os
from typing import Optional
from datetime import datetime

from sklearn.preprocessing import StandardScaler as scaler
from keras.models import Sequential
from keras.layers import Dense, GRU
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError, R2Score
from keras.optimizers import Adam
from keras.saving import load_model
from utils.plot_predictions import plot_line

from utils.load_dataset import load_non_missing_data, load_eval_columns
from utils.impute_fill import fill_missing_with_imputed_data

import warnings
warnings.filterwarnings("ignore")


class NeuralNetworkImputer:

    def __init__(self, dataset_cfg) -> None:
        self.data_name = dataset_cfg.data_name
        self.missing_data_path = dataset_cfg.missing_data_path
        self.time = datetime.now().strftime("%d%m%y_%H%M%S")
        self.columns = dataset_cfg.state_vector
        self.missing_cols = dataset_cfg.missing_state_vector
        self.model_directory = f"nnModels/{self.data_name}_model_{self.time}"

    def fit(self, target_col: str = None):
        print("Training Started...\n")
        #self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = load_non_missing_data(self.data_name, self.obs_size, splitted=True, predict_data=False, target_col_index=target_col)
        self.X_train, self.X_val, self.y_train, self.y_val = load_non_missing_data(self.missing_data_path, self.columns, self.missing_cols, target_col=target_col, splitted=True, test_data=False)
        #print(self.X_train.shape, self.X_val.shape, self.y_train.shape, self.y_val.shape)

        self.X_train_scaled = scaler().fit_transform(self.X_train)
        self.X_val_scaled = scaler().fit_transform(self.X_val)
        '''self.y_train_scaled = scaler().fit_transform(self.y_train)
        self.y_val_scaled = scaler().fit_transform(self.y_val)'''
        
        self.X_train_scaled = self.X_train_scaled.reshape((self.X_train_scaled.shape[0], 1, self.X_train_scaled.shape[1])) 
        self.X_val_scaled = self.X_val_scaled.reshape((self.X_val_scaled.shape[0], 1, self.X_val_scaled.shape[1]))
        #self.X_test  = self.X_test.values.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))
        #print(self.X_train.shape, self.X_val.shape, self.X_test.shape, self.y_train.shape, self.y_val.shape, self.y_test.shape)
        
        self.model = Sequential()
        self.model.add(GRU(64, input_shape=(1, self.X_train_scaled.shape[2])))
        #self.model.add(GRU(32, return_sequences=True))
        #self.model.add(GRU(16))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(1, activation='linear'))
        #print("Neural Netword based Imputation. \n")

        directory_path = f"{self.model_directory}/col_{target_col}/"

        #os.makedirs(directory_path, exist_ok=True)
        os.makedirs(f"{directory_path}", exist_ok=True)

        cp = ModelCheckpoint(directory_path, save_best_only=True)
        self.model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.01), metrics=[RootMeanSquaredError(), R2Score()])
        history = self.model.fit(self.X_train_scaled, self.y_train, validation_data=(self.X_val_scaled, self.y_val), epochs=20, callbacks=[cp], use_multiprocessing=True, workers=4)

        # Save the history to a file using pickle
        with open(f"{directory_path}training_history.pkl", 'wb') as file:
            pickle.dump(history.history, file)

        self.model.save(f"{directory_path}model.keras")

    def transform(self, target_col: str = None, saved_model_path: Optional[str] = None):
        print("Imputation Started...\n")
        try:
            test_data_df = load_non_missing_data(self.missing_data_path, self.columns, self.missing_cols, False, True, target_col)
            test_data_scaled = scaler().fit_transform(test_data_df)
            test_data_scaled = test_data_scaled.reshape(test_data_scaled.shape[0], 1, test_data_scaled.shape[1])

            if saved_model_path:
                print("Loading the saved model...\n")
                loaded_model = load_model(f"{saved_model_path}/col_{target_col}/model.keras")
                predicted_data = loaded_model.predict(test_data_scaled).flatten()

            elif saved_model_path is None:
                print("Loading the trained model...\n")
                predicted_data = self.model.predict(test_data_scaled).flatten()

        except Exception as ex:
            print("Error in Neural Nets transform:\n", ex)

        else:
            return predicted_data
    
    def fit_transform(self, target_col_ft: str = None):
        self.fit(target_col_ft)
        imputed_col = self.transform(target_col_ft)

        return imputed_col


    def compare(self, imp):
        og_col_1, og_col_2, imputed_col_1, imputed_col_2 = load_eval_columns(self.data_name, self.obs_size, "Imputation_results/Hopper-v4_w_imputed_nn_random.csv")

        plot_line(imputed_col_1[:50], og_col_1[:50])
        

    def fill_imputation(self, missing_dataframe, column, imputed_col):
        
        miss_col_rows = missing_dataframe.loc[:, column].isna()
        print(miss_col_rows.shape, imputed_col.shape)
        missing_dataframe.loc[miss_col_rows, column] = imputed_col

        return missing_dataframe
    
    def save_imputed_data(self, path, imputed_data):

        filled_data = fill_missing_with_imputed_data(self.missing_data_path, imputed_data)
        
        print("Saving the imputed data \n")
        filled_data.to_csv(f"{path}_nn.csv", index=False)
        print(f"Imputed Data saved at location \"{path}_nn.csv\"")
