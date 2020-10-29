import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection, linear_model

data = pd.read_csv('C:/Users/itc/PycharmProjects/scientificProject/data/car_prices.csv')

# fill null property by mean or median value
mileage = data.Mileage.mean()
data.Mileage = data.Mileage.fillna(mileage)

x = data[['Mileage', 'Age']]
y = data['Sell Price']

'''
plt.scatter(*data['Mileage', 'Age'], data['Sell Price'])
plt.xlabel('x axis for speed')
plt.ylabel('y axis for risk')
plt.title('Risk of speed')
plt.show()
'''

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3, random_state=2)
model = linear_model.LinearRegression()
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print('Model Accuracy', accuracy * 100, '%')


def predict_price():
    x = input('please enter Mileage: ')
    y = input('please enter Age: ')
    price = model.predict([[int(x), int(y)]])
    print("Predicted Price is ", price, " for milage and age----->", x, "   ", y)
    return predict_price()


predict_price()
