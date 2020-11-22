from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

dataset = pd.read_csv("C:/Users/itc/PycharmProjects/scientificProject/data/diabetes.csv")

x = dataset.drop('Outcome', axis=1)
y = dataset['Outcome']

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=150, batch_size=10, verbose=2)
predictions = model.predict(x[0])
print(predictions)