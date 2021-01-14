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

filename = '../../models/kmeans_model.pkl'
with open(filename, 'rb') as file:
    model = pk.load(file)

with open('model_params.json', 'r') as fp:
    kmeans_model1 = json.load(fp)

data=pd.read_csv('../../data/external/Mall_Customers.csv')
X_1 = data[["Age", "Spending_Score"]].values

pred = model.predict(X_1)
kmeans_model1["Centroids"] = model.cluster_centers_
kmeans_model1["Labels"] = model.labels_
#Visualizing all the clusters 

fig = plt.figure()
plt.scatter(x = 'Age' ,y = 'Spending_Score' , data = data , c = kmeans_model1["Labels"] , 
            s = 200, figure=fig)
plt.scatter(x = kmeans_model1["Centroids"][: , 0] , y =  kmeans_model1["Centroids"][: , 1] , s = 300 , c = 'red' , alpha = 0.5)
plt.ylabel('Spending Score (1-100)') , plt.xlabel('Age')
fig.savefig('../../reports/figures/segmentation.jpg')

