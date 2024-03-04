import os
import csv
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Evaluation:

    def scores(og_col, imputed_col, file_name, feature):
        mse = mean_squared_error(og_col, imputed_col)
        mae = mean_absolute_error(og_col, imputed_col)
        rmse = np.sqrt(mse)
        r2 = r2_score(og_col, imputed_col)

        file_path = "evals/evaluations.csv"
        flag = os.path.exists(file_path)

        time = datetime.now().strftime("%d/%m/%y %H:%M:%S")

        
        with open(file_path, mode="a", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            if flag:
                pass
            else:
                dw = csv.DictWriter(
                    csv_file,
                    delimiter=",",
                    fieldnames=[
                        "File",
                        "Time Stamp"
                        "Mean Squared Error (MSE)",
                        "Mean Absolute Error (MAE)",
                        "Root Mean Square Error (RMSE)",
                        "R2 Score",
                        "Feature"
                    ],
                )

                dw.writeheader()

            csv_writer.writerow([file_name, time, mse, mae, rmse, r2, feature])

        print(f"Evalution for {file_name} saved at {file_path}.")

    