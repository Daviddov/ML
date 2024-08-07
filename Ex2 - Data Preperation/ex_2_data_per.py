import pandas as pd
from sklearn.preprocessing import StandardScaler
fd = pd.read_csv('red_wine_quality.csv')
fd_init = fd.head(5)

fd['volatile acidity'] = fd['volatile acidity'].fillna(fd['volatile acidity'].mean())
fd['residual sugar'] = fd['residual sugar'].fillna(fd['residual sugar'].mean())
fd['sulphates'] = fd['sulphates'].fillna(fd['sulphates'].mean())
fd['free sulfur dioxide'] = fd['free sulfur dioxide'].fillna(fd['free sulfur dioxide'].mean())

stat = fd.describe()
missing_val = fd.isna().sum()

sc = StandardScaler()
fd.iloc[:,0:11] = sc.fit_transform(fd.iloc[:,0:11])

cors = fd.corr()
abs_corr = abs(cors['quality']).sort_values(ascending = False)
fd2 = fd[['alcohol','volatile acidity','quality']]

cors = fd2.corr()