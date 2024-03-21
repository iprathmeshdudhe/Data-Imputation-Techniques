import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError, R2Score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from imputers.Autoencoder import Autoencoders
from utils.impute_fill import fill_missing_with_imputed_data



class SSLImputer:
    def __init__(self, dataset_cfg) -> None:
        self.data_name = dataset_cfg.data_name
        self.time = datetime.now().strftime("%d%m%y_%H%M%S")
        self.columns = dataset_cfg.state_vector
        self.model_directory = f"AEModels/{self.data_name}_model_{self.time}"
        self.directory_path = f"{self.model_directory}/"
        self.s_i = SimpleImputer(strategy='mean')
        self.missing_data_path = dataset_cfg.missing_data_path
        self.imputed_path = dataset_cfg.imputed_data_path
               

    def build_model(self):
        self.scaler = StandardScaler()

        os.makedirs(f"{self.directory_path}", exist_ok=True)
        self.cp = ModelCheckpoint(self.directory_path, save_best_only=True)
         
        self.model = Autoencoders(input_dim=len(self.columns))
        self.model.compile(optimizer=SGD(learning_rate=0.01), loss=MeanSquaredError(), metrics=[RootMeanSquaredError(), R2Score()])

    def fill_missing_cells(self, missing_data_path, imputed_df):
        missing_data = pd.read_csv(missing_data_path, usecols=self.columns)
        #print(missing_data.head())
        missing_mask = np.where(np.isnan(missing_data), 1, 0)
        #print(missing_mask)
        filled_cells_df = imputed_df * missing_mask
        #print(filled_cells_df.head())

        filled_missing_cells_df = missing_data.fillna(filled_cells_df)
        #print(filled_missing_cells_df.head())

        return filled_missing_cells_df

    def fit(self, data: pd.DataFrame):
        print("Training Started...\n")
        train_data = data.dropna()
        train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

        self.scaler.fit(train_data)
        train_data_scaled = self.scaler.transform(train_data)
        val_data_scaled = self.scaler.transform(val_data)

        history = self.model.fit(train_data_scaled, train_data_scaled, validation_data=(val_data_scaled, val_data_scaled), epochs=20, verbose=1, callbacks=[self.cp], use_multiprocessing=True, workers=4)

        # Save the history to a file using pickle
        with open(f"{self.directory_path}training_history.pkl", 'wb') as file:
            pickle.dump(history.history, file)

        self.model.save(f"{self.directory_path}model.keras")

        print("Training Completed...\n")
    
    def transform(self, data: pd.DataFrame):
        print("Imputation Started...\n")
        
        # Data preprocessing
        test_data = self.s_i.fit_transform(data)
        test_data_scaled = self.scaler.transform(test_data)

        # Data Imputation
        imputed_data = self.model.predict(test_data_scaled)
        unscaled_imputed_data = self.scaler.inverse_transform(imputed_data)
        print("Imputation Completed...\n")

        imputed_df = pd.DataFrame(unscaled_imputed_data, columns=self.columns)
        #print(imputed_df.head())
        imputed_data_filled = self.fill_missing_cells(self.missing_data_path, imputed_df)
        filled_data = fill_missing_with_imputed_data(self.missing_data_path, imputed_data_filled)
        print("Saving the imputed data \n")
        filled_data.to_csv(f"{self.imputed_path}_ssl.csv", index=False)
        print(f"Imputed Data saved at location \"{self.imputed_path}_ssl.csv\"")
