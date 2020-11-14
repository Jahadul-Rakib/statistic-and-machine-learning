import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

data = pd.read_csv('C:/Users/itc/PycharmProjects/scientificProject/data/credit_card_taiwan.csv')
print('total row and column: ', data.shape)
print('total null value: ', data.isnull().sum())

mean_age = data.AGE.mean()
data.AGE = data.AGE.fillna(mean_age)

y = data['default.payment.next.month']
x = data.drop(['default.payment.next.month'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

# GaussianNB naive bays
model = GaussianNB()
model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)
print('Accuracy: ', accuracy)

# BernoulliNB naive bays
model1 = BernoulliNB()
model1.fit(x_train, y_train)

accuracy = model1.score(x_test, y_test)
print('Accuracy: ', accuracy)

# MultinomialNB naive bays
# if in dataset present negative value then its not working
'''
model2 = MultinomialNB()
model2.fit(x_train, y_train)

accuracy = model2.score(x_test, y_test)
print('Accuracy: ', accuracy)
'''
