import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

def clear_miss_data_columns(df):
    for col in df.columns:
        if df[col].isna().mean() > 0.1:
            df = df.drop(col, axis=1)
    return df

def clear_none_uniqu_columns(df):
    for col in df.columns:
        if df[col].nunique() == 1:
            df = df.drop(col, axis=1)
    return df

def fill_null_in_evg(df, col_name):
    df[col_name].fillna(df[col_name].mean(), inplace=True)
    return df

def replace_zero_to_null(df,col_name):
    df[col_name] = df[col_name].replace(0,np.nan)
    return df
    
def drop_contains_string(df):
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, str)).any():
            df = df.drop(col, axis=1)
            print(col)
    return df

def drop_low_correlation_columns(df, target_col, threshold=0.5):
    # Calculate the correlation matrix
    corr_matrix = df.corr()
    
    # Identify columns with correlation less than the threshold with the target column
    cols_to_drop = [col for col in df.columns if abs(corr_matrix[target_col][col]) < threshold and col != target_col]
    
    # Drop these columns
    df = df.drop(cols_to_drop, axis=1)
    
    return df

def drop_columns_with_low_target_correlation(df, target_col):
    # Calculate the correlation matrix
    corr_matrix = df.corr()
    
    # Sort columns by their absolute correlation with the target column in ascending order
    sorted_cols = corr_matrix[target_col].abs().sort_values().index
    
    # List to store columns to drop
    cols_to_drop = []
    
    # Iterate through sorted columns
    for col in sorted_cols:
        if col != target_col:
            col_corr_with_target = abs(corr_matrix[target_col][col])
            # Check if the column has a higher correlation with any other column than with the target column
            for other_col in df.columns:
                if other_col != target_col and other_col != col:
                    if abs(corr_matrix[col][other_col]) > col_corr_with_target:
                        cols_to_drop.append(col)
                        break
    
    # Drop the identified columns
    df = df.drop(cols_to_drop, axis=1)
    
    return df
# Read the data
df = pd.read_csv("Ex4_1.csv")

# Features and target variable
X = df.iloc[:, 1:5]
y = df.iloc[:, 0]

# Standardize the features
sc = StandardScaler()
X = sc.fit_transform(X)

# Compute correlation matrix
corr = df.corr()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Compute performance metrics
r2 = model.score(X_test, y_test)
mae = metrics.mean_absolute_error(y_test, y_pred)
coef = model.coef_

# Create DataFrame with predictions and actual values
df2 = pd.DataFrame({"predict": y_pred, "test": y_test})
df2.plot.bar()
scaled_data = sc.transform([[79,720,9.52,132]])
res = model.predict(scaled_data)