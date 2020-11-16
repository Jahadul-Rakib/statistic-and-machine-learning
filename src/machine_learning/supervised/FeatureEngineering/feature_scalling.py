import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

data = pd.read_csv('C:/Users/itc/PycharmProjects/scientificProject/data/feature_scaling.csv')

x = data[['Age', 'Salary']]

# Normalization
mms = MinMaxScaler(feature_range=(0, 1))
x_after_normalization_scaling = mms.fit_transform(x)

# Standardization
standardization = StandardScaler()
x_after_standard_scaling = standardization.fit_transform(x)