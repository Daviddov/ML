import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

def drop_low_correlation_columns(df, target_col, threshold=0.5):
    # Calculate the correlation matrix
    corr_matrix = df.corr()
    
    # Identify columns with correlation less than the threshold with the target column
    cols_to_drop = [col for col in df.columns if abs(corr_matrix[target_col][col]) < threshold and col != target_col]
    
    # Drop these columns
    df = df.drop(cols_to_drop, axis=1)
    
    return df

df = pd.read_csv("Boston_Housing.csv")
df = drop_low_correlation_columns(df,"MEDV",0.6)
corr = df.corr()
X = df.drop("MEDV",axis=1)
y = df["MEDV"]
sc = StandardScaler()
X = sc.fit_transform(X)
X = pd.DataFrame(X)

plt.scatter(X[0], y)
plt.scatter(X[1], y)

def poly_model(degree, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Predict on the train set
    y_train_pred = model.predict(X_train_poly)
    
    # Predict on the test set
    y_test_pred = model.predict(X_test_poly)

    rmse_train = metrics.root_mean_squared_error(y_train, y_train_pred)
    rmse_test = metrics.root_mean_squared_error(y_test, y_test_pred)
    
    r2_train = metrics.r2_score(y_train, y_train_pred)
    r2_test = metrics.r2_score(y_test, y_test_pred)

    # Print statements
    print('degree is:', degree)
    print('RMSE for train:', rmse_train)
    print('RMSE for test:', rmse_test)
    print('r2 score for train:', r2_train)
    print('r2 score for test:', r2_test)
    print('the rmse diff:', abs(rmse_train - rmse_test))
    print('the r2 diff:', abs(r2_train - r2_test))
    print(" ")

    return [rmse_train, rmse_test, r2_train, r2_test]

def find_best_degree(num=1):
    best_degree = -1
    r2_diff = float('inf')
    rmse_diff = float('inf')
    
    for i in range(1, num+1):
        rmse_train, rmse_test, r2_train, r2_test = poly_model(i, X,y)
        if rmse_train < 0 or rmse_test < 0 or r2_train < 0 or r2_test < 0:
            continue
        
        r2 = abs(r2_train - r2_test)
        rmse = abs(rmse_train - rmse_test)
        
        if r2 < r2_diff and rmse < rmse_diff:
            r2_diff = r2
            rmse_diff = rmse
            best_degree = i
            
    return best_degree

            
res = find_best_degree(8)

    
    
    



