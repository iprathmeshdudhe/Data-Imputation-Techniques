import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def write_missing_data_stats(df: pd.DataFrame, dataset_name: str):

    missing_info = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isna().sum(),
        'Missing_Percentage': df.isna().mean() * 100
    })

    missing_info.to_csv(f"stats/{dataset_name}_missing_stats.csv", index=False)


def plot_percentage_bar(dataframe: pd.DataFrame, dataset_name: str):
    # Assuming df is your DataFrame
    missing_percentage = dataframe.isnull().mean() * 100

    # Create a bar chart
    missing_percentage.plot(kind='bar')
    plt.title('Percentage of Missing Values by Column')
    plt.xlabel('Columns')
    plt.ylabel('Percentage of Missing Values')
    plt.tight_layout()
    plt.savefig(f"stats/{dataset_name}_percentage_stats.png")
    
    plt.show()
