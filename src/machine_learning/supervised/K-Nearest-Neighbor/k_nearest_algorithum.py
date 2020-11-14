from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# First Feature
weather = ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny',
           'Overcast', 'Overcast', 'Rainy']
# Second Feature
temp = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild']
# Label or target variable
play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']

# creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
weather_encoded = le.fit_transform(weather)
print(weather_encoded)

# converting string labels into numbers
temp_encoded = le.fit_transform(temp)
label = le.fit_transform(play)

# combining weather and temp into single listof tuples
features = list(zip(weather_encoded, temp_encoded))

x_train, x_test, y_train, y_test = train_test_split(features, play, test_size=0.20, random_state=1)

# n_estimator define how many subset tree will make
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)
print('Accuracy: ', accuracy)

predict = model.predict([[0, 2]])
print("Ans: ", predict)
