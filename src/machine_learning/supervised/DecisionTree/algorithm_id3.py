import pandas as pd
from sklearn import tree

data = pd.read_csv('C:/Users/itc/PycharmProjects/scientificProject/data/male_female_detect.csv')

x = data[['Height','Weight','Shoe_Size']]
y = data['Gender']

# Data classification by x,y in decision tree
classify_ = tree.DecisionTreeClassifier()
classify_.fit(x,y)


value = classify_.predict([[163,63,44]])
print('Gender: ', value)
