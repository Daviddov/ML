import pandas as pd



# pd1 = pd.DataFrame([[10,25,8],[9,31,10],[12,24,7]],columns=["Morning","Noon","Evening"],index=[2012,2014,2017])
# morning = pd1["Morning"]
# print( morning)

    
# temperatures_2017 = pd1.loc[2017]
# print(temperatures_2017)

# first_year = pd1.head(1)
# print(first_year["Evening"])


# temperatures_2014 = pd1.loc[2014]["Noon"]
# print(temperatures_2014)


df1 = pd.read_csv("C:\\Users\\Admin\\Downloads\\חומרי לימוד\\Ex2.csv")
df2 = pd.read_csv("C:\\Users\\Admin\\Downloads\\חומרי לימוד\\Ex2.csv")
df3  = pd.concat([df1,df2],axis=0)
df3 = df3.reset_index(drop =True)
df3 = df3.drop("Price", axis=1)
df3 = df3.drop([1,3], axis=0)
df3.reset_index(drop =True, inplace =True)




