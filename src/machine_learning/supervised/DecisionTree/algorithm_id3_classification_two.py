import pandas as pd
from sklearn import tree, preprocessing

data = pd.read_csv('C:/Users/itc/PycharmProjects/scientificProject/data/shop_data.csv')

y = data['buys']
x = data.drop('buys',axis=1)

# Convert String Data to Numeric form
lvl_encoder = preprocessing.LabelEncoder()
x = x.apply(lvl_encoder.fit_transform)

# fit data
classify = tree.DecisionTreeClassifier()
classify.fit(x, y)

y_predict = classify.predict([[1,2,0,1]])
print("Do you buy: ", y_predict)