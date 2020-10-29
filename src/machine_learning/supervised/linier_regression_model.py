import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

'''
* prediction comes upon independent variable for dependent variable.
* independent variable(feature), dependent variable(input).
* pandas library for reading data from data source.
'''
# read file from data source
data_set = pd.read_csv('C:/Users/itc/PycharmProjects/scientificProject/data/home_price.csv')
# show first 5 rows without pass parameter
data_set.head()
# show total row and column (data_set.shape)

# checking any field is null or not
null_notnull = data_set.isnull().any()
print(null_notnull)

# provide total null
total_null = data_set.isnull().sum()
print(total_null)

# Independent variable always become 2 dimensional
x = data_set[["area"]]
y = data_set["price"]

# Show plot
plt.scatter(data_set["area"], data_set["price"])
plt.xlabel('X axis Area')
plt.ylabel('Y axis Price')
plt.title('Home Prices In Dhaka')
plt.show()

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30, random_state=2)

regration = LinearRegression()
regration.fit(xtrain, ytrain)
regration.predict(xtest)

# brest feed line
plt.scatter(data_set["area"], data_set["price"])
plt.xlabel('X axis Area')
plt.ylabel('Y axis Price')
plt.title('Home Prices In Dhaka')
plt.plot(data_set.area, regration.predict(data_set[['area']]))
plt.show()

# provide predicted result for different area
price = regration.predict([[21000]])
print(price)

price = regration.predict([[100]])
print(price)

price = regration.predict([[10]])
print(price)