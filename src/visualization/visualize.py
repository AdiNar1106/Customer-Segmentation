import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv('../../data/external/Mall_Customers.csv')

plt.style.use('fivethirtyeight')

# Distribution Plotting
fig1 = plt.figure(figsize=(8, 5))
sns.distplot(data["Age"])
plt.xlabel("Age")
plt.title("Distribution of Age")
fig1.savefig('../../reports/figures/age_dist.png')

fig2 = plt.figure(figsize=(8, 5))
sns.distplot(data["Annual_Income"])
plt.xlabel("Annual Income (k$)")
plt.title("Distribution of Annual Income")
fig2.savefig('../../reports/figures/income_dist.png')

fig3 = plt.figure(figsize=(8, 5))
sns.distplot(data["Spending_Score"])
plt.xlabel("Spending Score")
plt.title("Distribution of Spending Score")
fig3.savefig('../../reports/figures/score_dist.png')

# Gender Bar Plot
fig4 = plt.figure(1 , figsize = (15 , 5))
sns.countplot(y = 'Gender' , data = data)
fig4.savefig('../../reports/figures/gender_hist.png')

# Variable Correlations
fig5 = plt.figure(figsize=(8, 5))
sns.pairplot(data[["Age", "Annual_Income", "Spending_Score"]])
fig5.savefig('../../reports/figures/variable_correlations.png')

# Scatter plot of Age and Spending Score grouped by gender
fig6 = plt.figure(figsize=(8, 5))
sns.scatterplot(x="Age", y="Spending_Score", data=data, hue="Gender")
fig6.savefig('../../reports/figures/Score_vs_Age.png')

# Scatter plot of Income and Spending Score grouped by gender
fig7 = plt.figure(figsize=(8, 5))
sns.scatterplot(x="Annual_Income", y="Spending_Score", data=data, hue="Gender")
fig7.savefig('../../reports/figures/Score_vs_Income.png')


