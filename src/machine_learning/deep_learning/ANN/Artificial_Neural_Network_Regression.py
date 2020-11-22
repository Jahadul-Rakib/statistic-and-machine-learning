import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

dataset = pd.read_csv("C:/Users/itc/PycharmProjects/scientificProject/data/diabetes.csv")

x = dataset.drop('Outcome', axis=1)
y = dataset['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

# Transform value range between 0 and 1
scaler = MinMaxScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

model = Sequential()
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=150)  # , batch_size=10, verbose=2

model.save('C:/Users/itc/PycharmProjects/scientificProject/models/load.h5')

loss_history = pd.DataFrame(model.history.history)
loss_history.plot()

loss = model.evaluate(x_test, y_test, verbose=0)
print("Loss: ", loss)

# test_prediction = model.predict(x_test)
# predicted = pd.Series(test_prediction.reshape(300,))
#
# true_Value = pd.DataFrame(y_test, columns=["True Value"])
# predicted_dataset = pd.concat([true_Value, predicted], axis=1)
#
# print("Prediction: ", )

# predictions = model.predict(x[0])
# print(predictions)
loaded_model = load_model('/models/load.h5')
result = loaded_model.predict([[7,62,78,0,0,32.6,0.391,41]])
print("Result: ", result)