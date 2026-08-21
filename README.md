# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. Import the required libraries.
2. Load the Mall Customer dataset.
3. Select Annual Income and Spending Score as features.
4. Apply K-Means clustering with **5 clusters**.
5. Plot the clusters and their centroids.
 

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

# Load the dataset
data = pd.read_csv("Mall_Customers.csv")

# Select Annual Income and Spending Score columns
X = data.iloc[:, [3, 4]].values

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=100)

# Plot the centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    marker='X'
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")

plt.show()
```

## Output:
<img width="567" height="419" alt="image" src="https://github.com/user-attachments/assets/a09e6a34-fdc1-464f-bfc7-354fccd243be" />



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
