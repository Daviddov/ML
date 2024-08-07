import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def drop_low_correlation_columns(df, target_col, threshold=0.5):
    # Calculate the correlation matrix
    corr_matrix = df.corr()
    
    # Identify columns with correlation less than the threshold with the target column
    cols_to_drop = [col for col in df.columns if abs(corr_matrix[target_col][col]) < threshold and col != target_col]
    
    # Drop these columns
    df = df.drop(cols_to_drop, axis=1)
    
    return df

df = pd.read_csv("Ex5.csv")

# corr = df.corr()
# df = drop_low_correlation_columns(df, "Petrol_Consumption", 0.4)

X = df.drop("Petrol_Consumption", axis=1)


y = df["Petrol_Consumption"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressor = DecisionTreeRegressor()

regressor.fit(X_train, y_train)

# y_pred = regressor.predict([[7.2,0.47]])

r2 = regressor.score(X_test, y_test)