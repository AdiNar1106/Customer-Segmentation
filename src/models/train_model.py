import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.cluster import KMeans
import warnings
import os
import pickle as pk
warnings.filterwarnings("ignore")
import json


data=pd.read_csv('../../data/external/Mall_Customers.csv')
X_1 = data[["Age", "Spending_Score"]].values

kmeans_model1 = {}

def get_inertia(X, end):
    inertia = []
    for i in range(1,end):
        model = KMeans(n_clusters= i, init='k-means++', random_state=0)
        model.fit(X)
        inertia.append(model.inertia_)
    return inertia

fig = plt.figure()
plt.plot(range(1,20), get_inertia(X_1, 20), marker='o', color='b')
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('inertia')
fig.savefig('../../reports/figures/inertia.jpg')

kmeans_model1["Inertia"] = get_inertia(X_1, 20)

model = KMeans(n_clusters= 4, init='k-means++', random_state=0)
model.fit(X_1)

filename = 'kmeans_model.pkl'
pk.dump(model, open(filename, 'wb'))

with open('model_params.json', 'w') as fp:
    json.dump(kmeans_model1, fp)
