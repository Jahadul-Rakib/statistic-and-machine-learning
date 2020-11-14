import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('C:/Users/itc/PycharmProjects/scientificProject/data/cardio_train.csv')
print('total row and column: ', data.shape)
print('total null value: ', data.isnull().sum())

# we can drop ID column
data = data.drop('id', axis=1)

y = data['cardio']
x = data.drop('cardio', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=5)

# n_estimator define how many subset tree will make
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)
print('Accuracy: ', accuracy)
