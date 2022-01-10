import pandas as pd


data=pd.read_csv('laptop_data.csv')
data.drop('Unnamed: 0', inplace=True, axis=1)
# print(data)

#splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)