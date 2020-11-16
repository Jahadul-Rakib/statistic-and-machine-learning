# In here No Label data will be use
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.cluster import KMeans

'''
1. every data placed on a cluster.
2. every cluster has a centroid (mean value of all data in a cluster)
3. when comes new data then calculate distance from all centroid.
4. comers placed on that cluster which centroid value distance is low from comers point.
'''

data = pd.read_csv('C:/Users/itc/PycharmProjects/scientificProject/data/Mall_Customers.csv')

data.rename(columns={'Gender': 'gender', 'Age': 'age', 'AnnualIncome(k$)': 'income',
                     'SpendingScore(1-100)': 'spend'}, inplace=True)

sbn.pairplot(data[['age', 'income', 'spend']])
plt.show()

k_means = KMeans(n_clusters=5)
k_means.fit(data[['income', 'spend']])
cluster_centroid = k_means.cluster_centers_

data['income_cluster'] = k_means.labels_
print("Total income cluster value: ", data['income_cluster'].value_counts())

sbn.scatterplot(x=data['income'], y=data['spend'], hue=data['income_cluster'], data=data)
plt.show()


k_means1 = KMeans(n_clusters=5)
k_means1.fit(data[['age', 'spend']])  #
cluster_centroid1 = k_means1.cluster_centers_

data['age_cluster'] = k_means1.labels_
print("Total income cluster value: ", data['age_cluster'].value_counts())
sbn.scatterplot(x=data['age'], y=data['spend'], hue=data['age_cluster'], data=data)
plt.show()