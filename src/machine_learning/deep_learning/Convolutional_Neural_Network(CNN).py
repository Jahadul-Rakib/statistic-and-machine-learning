import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('C:/Users/itc/PycharmProjects/scientificProject/data/data.csv')
data.set_index('id', inplace=True)

print(data.isnull().sum())

data['diagnosis'] = data['diagnosis'].map(lambda x: 1 if (x == 'M') else 0)

x = data.drop('diagnosis', axis=1)
y = data['diagnosis']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=5, stratify=y)
