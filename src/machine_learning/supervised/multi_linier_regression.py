import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('C:/Users/itc/PycharmProjects/scientificProject/data/marketing_profit.csv')
# get total null value by column
print(data.isnull().sum())

# handle null value for Administration column
mean_value1 = data.Administration.mean()
data.Administration = data.Administration.fillna(mean_value1)

# handle null value for Marketing_Spend column
mean_value2 = data.Marketing_Spend.mean()
data.Marketing_Spend = data.Marketing_Spend.fillna(mean_value2)

# handle null value for Transport column
mean_value3 = data.Transport.mean()
data.Transport = data.Transport.fillna(mean_value3)

x = data.drop(['Profit'], axis=1)
y = data['Profit']

# convert string column to one-hot encoding
area = pd.get_dummies(x['Area'], drop_first=True)
# drop area column from x data set
x = x.drop('Area', axis=1)
# concat area with main x dataset
x = pd.concat([x, area], axis=1)

# split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=12)

model = LinearRegression()
model.fit(x_train, y_train)

# model accurecy score
accurecy = model.score(x_test, y_test)
print('Model Accurecy ', accurecy)


def predict():
    maketing = input('please enter speeds Marketing_Spend: ')
    administration = input('please enter speeds Administration: ')
    transport = input('please enter speeds Transport: ')
    dhaka = input('please enter speeds Area Dhaka: ')
    rangpur = input('please enter speeds Area Rangpur: ')

    # Predict model for user provided value
    profit = model.predict([[int(maketing), int(administration), int(transport), int(dhaka), int(rangpur)]])
    print("Predicted Data is ", profit)
    return predict()


predict()
