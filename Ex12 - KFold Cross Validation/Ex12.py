
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("Ex12.csv")

X = df.drop(["Outcome"], axis=1)
y = df["Outcome"]

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# kfold = KFold(n_splits=10)
model = KNeighborsClassifier()
# results_kfold = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
# mean_accuracy = results_kfold.mean() 
# 0.7265550239234451
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# param_grid ={
#         'n_neighbors' : [2,4,5,7,9],
#        'weights': ['uinform','distance']
#     }
# gs = GridSearchCV(model, param_grid, cv = 10, scoring= 'accuracy')
# gs.fit(X_train, y_train)
# best_score = gs.best_score_
# best_parms = gs.best_params_
# # 0.75

# model = LogisticRegression()
# param_grid = {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100],
#     'penalty': ['l1', 'l2'],
#     'solver': ['liblinear', 'saga']
# }
# gs = GridSearchCV(model, param_grid, cv = 10, scoring= 'accuracy')
# gs.fit(X_train, y_train)
# best_score = gs.best_score_
# best_parms = gs.best_params_
# # 0.77

model = RandomForestClassifier(random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [10],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}
gs = GridSearchCV(model, param_grid, cv = 20, scoring= 'accuracy')
gs.fit(X_train, y_train)
best_score = gs.best_score_
best_parms = gs.best_params_
# 0.768
