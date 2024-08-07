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


df = pd.read_csv("Ex3.csv")
# df.plot.scatter(x='MinTemp', y='MaxTemp', color='gray')
# df.plot.line()
df = fill_null_in_evg(df, 'MIN')
df = fill_null_in_evg(df, 'MAX')
df = fill_null_in_evg(df, 'MEA')

# df["Snowfall"].fillna(0, inplace=True)
# df["SNF"].fillna(0, inplace=True)

df = clear_miss_data_columns(df)
df = df.drop("PRCP", axis=1)
df = df.drop("MeanTemp", axis=1)
df = df.drop("MAX", axis=1)
df = df.drop("MIN", axis=1)
df = df.drop("MEA", axis=1)
# df = df.drop("Date", axis=1)
df = clear_none_uniqu_columns(df)
df = drop_contains_string(df)
stat =  df.describe()
missing_values = df.isna().sum()

df = drop_low_correlation_columns(df,"MaxTemp")
# df = drop_columns_with_low_target_correlation(df, "MaxTemp")
df = df.drop_duplicates()

co = df.corr()
X = df.drop("MaxTemp",axis=1)
y = df["MaxTemp"]

# sc = StandardScaler()
# X = sc.fit_transform(X)

train_X,test_X,train_y,test_y = train_test_split(X,y, test_size=0.2)

model = LinearRegression()
model.fit(train_X,train_y)
y_pred = model.predict(test_X)
# df.plot.scatter(x=test_X, y=test_y, color='gray')
# df.plot(x=test_X, y=predict, color='red')
# coef = model.coef_
# inter = model.intercept_
r2 = model.score(test_X, test_y)
mae = metrics.mean_absolute_error(test_y, y_pred)
plt.scatter(test_X, test_y,color="red")
plt.plot(test_X, y_pred,color="blue")
res = model.predict([[22.6]])