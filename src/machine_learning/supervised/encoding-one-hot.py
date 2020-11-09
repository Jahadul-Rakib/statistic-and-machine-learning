import pandas as pd
from sklearn import linear_model, model_selection

data = pd.read_csv('C:/Users/itc/PycharmProjects/scientificProject/data/carprices.csv')

# Convert String to Numeric form
dummies = pd.get_dummies(data['Car Model'])

# Merged two DataSet
merged = pd.concat([data, dummies], axis='columns')

# Now Drop String Column
final_data_set = merged.drop(['Car Model'], axis='columns')

y = final_data_set['Sell Price']
x = final_data_set.drop(['Sell Price'], axis='columns')

# split dataset
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3, random_state=2)

model = linear_model.LinearRegression()
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print('Model Accuracy: ', accuracy)

price = model.predict([[2000,12,0,0,1,0]])
print('Price: ', price)
