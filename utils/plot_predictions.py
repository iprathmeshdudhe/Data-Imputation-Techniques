import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns

def plot_line(predictions, actual):
    # Create a DataFrame
    df = pd.DataFrame({'Actual': actual, 'Predictions': predictions})

    # Set the style and context for Seaborn
    sns.set(style="darkgrid")
    sns.set_context("talk")

    # Plotting the data using Seaborn
    plt.figure(figsize=(8, 6))
    ax = sns.lineplot(data=df)
    
    # Customize the plot
    ax.set_title("Actual and Predicted Comparison")
    ax.set(xlabel="Timesteps", ylabel="Observations")
    plt.legend()  # You can adjust the location as needed
    
    # Show the plot
    plt.show()
