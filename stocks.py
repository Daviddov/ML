import numpy as np # linear algebra
import pandas as pd 

import matplotlib.pyplot as plt
from datetime import datetime

def drop_low_correlation_columns(df, target_col, threshold=0.5):
    # Calculate the correlation matrix
    corr_matrix = df.corr()
    
    # Identify columns with correlation less than the threshold with the target column
    cols_to_drop = [col for col in df.columns if abs(corr_matrix[target_col][col]) < threshold and col != target_col]
    
    # Drop these columns
    df = df.drop(cols_to_drop, axis=1)
    
    return df

df = pd.read_csv("Israel_Stocks.csv")

init = df.head(5)
col = df.columns
stocks_names=list(df["Symbol"].str.split(' ', expand=True).stack().unique())
time_format = '%d/%m/%y'
for z in stocks_names:
    i=df[df["Symbol"]==z]
    time = [datetime.strptime(x, time_format) for x in i['Date']]
    plt.plot(time,i['Closing Price (0.01 NIS)'])
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.title(z)
    plt.show()
