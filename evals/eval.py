import os
import csv
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Evaluation:

    def scores(og_col, imputed_col, file_name):
        mse = mean_squared_error(og_col, imputed_col)
        mae = mean_absolute_error(og_col, imputed_col)
        rmse = np.sqrt(mse)
        r2 = r2_score(og_col, imputed_col)

        file_path = "evals/evaluations.csv"
        flag = os.path.exists(file_path)

        
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
                        "Mean Squared Error (MSE)",
                        "Mean Absolute Error (MAE)",
                        "Root Mean Square Error (RMSE)",
                        "R2 Score",
                    ],
                )

                dw.writeheader()

            csv_writer.writerow([file_name, mse, mae, rmse, r2])

        print(f"Evalution for {file_name} saved at {file_path}.")

    