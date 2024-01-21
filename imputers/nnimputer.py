import pickle
import os
from datetime import datetime
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

    def __init__(self, dataset: str, columns: list, missing_type: str, miss_cols: list) -> None:
        self.data_name = dataset
        self.time = datetime.now().strftime("%d%m%y_%H%M%S")
        self.m_type = missing_type
        self.columns = columns
        self.missing_cols = miss_cols

    def fit(self, target_col: str = None):

        #self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = load_non_missing_data(self.data_name, self.obs_size, splitted=True, predict_data=False, target_col_index=target_col)
        self.X_train, self.X_val, self.y_train, self.y_val = load_non_missing_data(self.data_name, self.columns, self.missing_cols, target_col=target_col, splitted=True, predict_data=False)
        #print(self.X_train.shape, self.X_val.shape, self.y_train.shape, self.y_val.shape)
        
        self.X_train = self.X_train.values.reshape((self.X_train.shape[0], 1, self.X_train.shape[1])) 
        self.X_val = self.X_val.values.reshape((self.X_val.shape[0], 1, self.X_val.shape[1]))
        #self.X_test  = self.X_test.values.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))
        #print(self.X_train.shape, self.X_val.shape, self.X_test.shape, self.y_train.shape, self.y_val.shape, self.y_test.shape)
        
        self.model = Sequential()
        #self.model.add(InputLayer((1, self.X_train.shape[2])))
        self.model.add(GRU(64, input_shape=(1, self.X_train.shape[2])))
        #self.model.add(GRU(32, return_sequences=True))
        #self.model.add(GRU(16))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(1, activation='linear'))
        print("Neural Netword based Imputation. \n")

        directory_path = f"nnModels/{self.data_name}_{self.m_type}_model_{self.time}/col_{target_col}/"
        os.makedirs(directory_path, exist_ok=True)

        cp = ModelCheckpoint(directory_path, save_best_only=True)
        self.model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.01), metrics=[RootMeanSquaredError(), R2Score()])
        history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs=10, callbacks=[cp])

        # Save the history to a file using pickle
        with open(f"{directory_path}training_history.pkl", 'wb') as file:
            pickle.dump(history.history, file)

        self.model.save(f"{directory_path}model.keras")

    def predict(self):
        #imputations = self.model.predict(self.X_test).flatten()
        #self.test_results = pd.DataFrame(data={'Test Predictions':imputations, 'Actuals':self.y_test})
        pass

    def transform(self, target_col: str = None):
        test_data = load_non_missing_data(self.data_name, self.columns, self.missing_cols, False, True, target_col)
        test_data = test_data.values.reshape(test_data.shape[0], 1, test_data.shape[1])

        predicted_data = self.model.predict(test_data).flatten()

        return predicted_data

    def compare(self, imp):
        og_col_1, og_col_2, imputed_col_1, imputed_col_2 = load_eval_columns(self.data_name, self.obs_size, "Imputation_results/Hopper-v4_w_imputed_nn_random.csv")

        plot_line(imp[:50], og_col_1[:50])
        

    def save_imputation(self, missing_dataframe, column, imputed_col):
        
        miss_col_rows = missing_dataframe.loc[:, column].isna()
        print(miss_col_rows.shape, imputed_col.shape)
        missing_dataframe.loc[miss_col_rows, column] = imputed_col

        return missing_dataframe

