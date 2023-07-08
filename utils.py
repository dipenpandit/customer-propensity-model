import numpy as np
import pandas as pandas
import matplotlib.pyplot as plt
import joblib

# Convert csv to input array
def convert(df_input):
    x_input = np.array(df_input.drop('Name', axis=1))
    return x_input

# Load model
def predict(x_input):
    with open('savedmodel.joblib', 'rb') as f:
        model = joblib.load(f)
    yhat = model.predict(x_input)
    return yhat

# Filter the dataframe containing the customers who are likely to make purchase
def filter(yhat, df):
    index = np.where(yhat == 1)
    out_df = df.loc[index]
    return out_df

# Plot the pie chart showing the number of people who made the purchased
def plot_pie(y):
    labels = ['Not Purchased', 'Purchased']

    # Count the occurrences of zeros and ones
    zero_count = np.count_nonzero(y==0)
    ones_count = np.count_nonzero(y==1)

    counts = [zero_count, ones_count]
    plt.figure(figsize=(7,7))  
    # Plot the pie chart
    plt.pie(counts, labels=labels, autopct='%1.1f%%')
    plt.legend()
    # Set aspect ratio to be equal so that pie is drawn as a circle
    plt.axis('equal')
    plt.show()

