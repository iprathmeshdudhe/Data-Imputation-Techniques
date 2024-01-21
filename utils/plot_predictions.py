import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns

def plot_line(predictions, actual, dataset_name: str, imputer: str, variable: str):
    # Create a DataFrame
    df = pd.DataFrame({'Actual': actual, 'Predictions': predictions})

    # Set the style and context for Seaborn
    sns.set(style="darkgrid")
    sns.set_context("talk")

    # Plotting the data using Seaborn
    plt.figure(figsize=(8, 6))
    ax = sns.lineplot(data=df)
    
    # Customize the plot
    ax.set_title(f"Actual vs Predicted - {dataset_name} & {imputer} - {variable}")
    ax.set(xlabel="Timesteps", ylabel="Observations")
    plt.legend()  # You can adjust the location as needed
    plt.savefig(f"imputation_plots/{dataset_name}_{imputer}_{variable}.png")
    # Show the plot
    #plt.show()
