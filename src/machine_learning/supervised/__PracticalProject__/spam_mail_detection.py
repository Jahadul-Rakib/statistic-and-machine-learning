import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv('C:/Users/itc/PycharmProjects/scientificProject/data/emails.csv')

# Drop Duplicate value
data = data.drop_duplicates(inplace=True)

x = data.text.values
y = data.spam.values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

# convert string to column by count vectorizer
cv = CountVectorizer()
x_train = cv.fit(x_train)
x_test = cv.fit(x_test)


# fit model
model = MultinomialNB()
model.fit(x_train,y_train)

accuracy = model.score(x_test, y_test)
print('Accuracy: ',accuracy)

# Take email text and transform it by label encoder
mail =["Hello Man, I want to get Your contact Info.", "Yot can send me your personal Info."]
mail_transform = cv.fit(mail)

value = model.predict(mail_transform)
print('Result: ',value)