import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('C:/Users/itc/PycharmProjects/scientificProject/data/car driving risk analysis.csv')
x = data[['speed']]
y = data['risk']

plt.scatter(data['speed'], data['risk'])
plt.xlabel('x axis for speed')
plt.ylabel('y axis for risk')
plt.title('Risk of speed')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.30, random_state=2)

reg_model = LinearRegression()
reg_model.fit(x_train, y_train)
accurecy = reg_model.score(x_test, y_test)
print('Model Accurecy ', accurecy)

plt.scatter(data['speed'], data['risk'])
plt.xlabel('x axis for speed')
plt.ylabel('y axis for risk')
plt.title('Risk of speed')
plt.plot(data.speed, reg_model.predict(data[['speed']]))
plt.show()


def predict_risk():
    x = input('please enter speeds: ')
    risk = reg_model.predict([[int(x)]])
    print("Predicted Risk is ", risk, " for speed ", x)
    return predict_risk()


predict_risk()
