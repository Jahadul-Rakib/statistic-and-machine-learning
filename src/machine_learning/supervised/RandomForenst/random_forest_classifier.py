import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('C:/Users/itc/PycharmProjects/scientificProject/data/credit_card_taiwan.csv')
print('total row and column: ', data.shape)
print('total null value: ', data.isnull().sum())

mean_age = data.AGE.mean()
data.AGE = data.AGE.fillna(mean_age)

y = data['default.payment.next.month']
x = data.drop(['default.payment.next.month'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

# n_estimator define how many subset tree will make
model = RandomForestClassifier(n_estimators=200)
model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)
print('Accuracy: ', accuracy)
