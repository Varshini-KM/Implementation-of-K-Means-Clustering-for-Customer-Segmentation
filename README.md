# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook


### ALGORITHM:

1. Import the required libraries and load the Mall Customer dataset.
2. Select **Annual Income** and **Spending Score** as the features for clustering.
3. Create the K-Means model with **5 clusters**.
4. Train the model and assign each customer to a cluster.
5. Plot the clusters along with their centroids.

 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: VARSHINI K M
RegisterNumber: 212225240179
*/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

X = data.iloc[:, [3, 4]].values

kmeans = KMeans(n_clusters=5, random_state=0)

y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=100)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    marker="X"
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")

plt.show()
```

## Output:
<img width="512" height="388" alt="Screenshot 2026-08-28 145446" src="https://github.com/user-attachments/assets/fdc16e11-eae7-4982-adff-68aa506b27ae" />




## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
