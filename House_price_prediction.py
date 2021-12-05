import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt

train_data_path=r"C:\Users\ajay\Desktop\HousePricePrediction\Data_Set\train.csv"      #Assigning the path of data set to a variable
test_data_path=r"C:\Users\ajay\Desktop\HousePricePrediction\Data_Set\test.csv"

df_train=pd.read_csv(train_data_path)            #Function to read csv files
df_test=pd.read_csv(test_data_path)

'''print(df_test.shape)                          #for checking rows and columns in our data set
print(df_train.shape)'''

'''print(df_train.head())                           #for displaying rows and columns in our data set
print(df_test.head())'''

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)

#print(df_train.head())                          #for displaying all rows and columns in our data set
#print(df_test.head())

df=pd.concat([df_test,df_train])                 #Combining both set of data 
#print("Shape of Integrated data :",df.shape)  

print(df.info())                                #Printing the info of the data (Like:- Its no.of null and non-null values,type ,name etc...)

int_features=df.select_dtypes(include=["int64"]).columns     #Extracting Integer types of features
print("Total no. of Integer features : ",int_features.shape[0])
print("Integer features names : ",int_features.tolist())

float_features=df.select_dtypes(include=["float64"]).columns     #Extracting Flaot types of features
print("Total no. of Float features : ",float_features.shape[0])
print("Float features names : ",float_features.tolist())

ctgry_features=df.select_dtypes(include=["object"]).columns     #Extracting Catergorical types of features
print("Total no. of Categorical features : ",ctgry_features.shape[0])
print("Catergorical features names : ",ctgry_features.tolist())

#                      MOST NULL VALUE FEATURE

#Alley
#FireplaceQu  
#PoolQC  
#Fence 
#MiscFeature

