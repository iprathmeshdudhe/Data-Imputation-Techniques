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


def corr_mat(data: pd.DataFrame, dataset_name: str):
    # Calculate the correlation matrix
    correlation_matrix = data.corr()

    # Plot the correlation matrix as a heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Between Observations')
    #plt.savefig(f"stats/{dataset_name}_corr_mat.png", bbox_inches='tight')
    plt.show()


def observation_trend(col: pd.Series):
    # Set the style and context for Seaborn
    sns.set(style="darkgrid")
    sns.set_context("talk")

    # Plotting the data using Seaborn
    plt.figure(figsize=(18, 6))
    ax = sns.lineplot(data=col)

    # Customize the plot
    ax.set_title(f"Observation Trend Over Time - {col.name}")
    ax.set(xlabel="Timesteps", ylabel="Observations")
    plt.savefig(f"stats/observation_trend_{col.name}.png", bbox_inches='tight')


data, name = pd.read_csv("dataset/mimiciv_obs_data_not_null.csv"), "mimiciv"
#write_missing_data_stats(data, name)
corr_mat(data, name)
