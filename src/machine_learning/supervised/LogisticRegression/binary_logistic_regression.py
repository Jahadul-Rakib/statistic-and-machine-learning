# when outcome comes only two value like true false then use it.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('C:/Users/itc/PycharmProjects/scientificProject/data/merit_status.csv')
# Checking Null Value
print(data.isnull().sum())

# When in column contain 2 types data we should fill nan position by median
median = data.status.median()
data.status = data.status.fillna(median)

# count how many people are married or not
print('Married: ',data.status.value_counts())

x = data[['age']]
y = data['status']

x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.30, random_state=1)

model = LogisticRegression()
model.fit(x_train,y_train)
accuracy = model.score(x_test, y_test)
print('Model Accuracy: ', accuracy)
# predict probability
probability = model.predict_proba(x_test)
print('Predict Probability: ',probability)
print('Predict Result For X_TEST: ',model.predict(x_test))

print('Predicted Result: ', model.predict([[33]]))