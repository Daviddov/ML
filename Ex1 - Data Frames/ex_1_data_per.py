import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("Ex1_Data1.csv")
df2 = pd.read_csv("Ex1_Data2.csv")
# df3 = pd.concat([df,df2] ,axis=1) 
df3 = pd.merge(df,df2,on='PlayerID')
data_init = df3.head(5)
stat = df3.describe()
df4 = df3.drop_duplicates()

df4['height (m)'] = df4['height (m)'].replace(0,np.nan)
missing_values = df4.isna().sum()
df4['avg fouls/min'] = df4['avg fouls/min'].fillna(df4['avg fouls/min'].mean())
df4['avg rebounds/min'] = df4['avg rebounds/min'].fillna(df4['avg rebounds/min'].mean())
df4['height (m)'] = df4['height (m)'].fillna(df4['height (m)'].mean())
missing_values = df4.isna().sum()
stat = df4.describe()



