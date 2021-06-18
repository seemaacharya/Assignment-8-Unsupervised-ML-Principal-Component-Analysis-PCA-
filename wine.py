# -*- coding: utf-8 -*-
"""
Created on Fri May 21 16:12:57 2021

@author: DELL
"""

#Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

#loading the dataset
wine = pd.read_csv("wine.csv")
wine.head()
#Describing the data to know the std dev and others
wine.describe()

#considering only the numerical data
wine.data = wine.iloc[:,1:]
wine.data.head()

#Normalizing the numerical data
wine_normal = scale(wine.data)
wine_normal

#using the PCA function of sklearn library and save in pcr var upto PC1-PC3
pca = PCA(n_components =3)
pca_values = pca.fit_transform(wine_normal)
pca_values

#the amount of variance that each PCA explains is
var = pca.explained_variance_ratio_
var

#showing result of PC1
pca.components_[0]

#cumulative variance
var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1

#Among 12 columns can we dimension reduction upto 8 columns nearly 90%(PC1)

#variance plot for PCA components obtained
#scree plot
plt.figure(figsize=(10, 4))
plt.plot(var1,color="red")

#plot between PCA1 and PCA2
x = pca_values[:,0]
y = pca_values[:,1]
plt.plot(x,y,"ro")

#Hierarchial clustering
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(wine_normal)
X_scaled_df = pd.DataFrame(X_scaled)
X_scaled_df.head()

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
#single linkage
single_linkage = linkage(X_scaled_df, method= "single", metric ="euclidean")
dendrogram(single_linkage)
plt.show()
#complete linkage
complete_linkage = linkage(X_scaled_df, method= "complete", metric ="euclidean")
dendrogram(complete_linkage)
plt.show()

#3 clusters using single linkage
from scipy.cluster.hierarchy import cut_tree
single_cluster_labels = cut_tree(single_linkage, n_clusters=3).reshape(-1,)
single_cluster_labels
#the single clustering does not perform well in generating the clusters. Hence, we go for complete linkage

# 3 clusters using complete linkage
complete_cluster_labels = cut_tree(complete_linkage, n_clusters=3).reshape(-1,)
complete_cluster_labels

#plot
plt.figure(figsize=(8, 4))
plt.title('Hierarchial clustering dendrogram')
plt.xlabel("Index")
plt.ylabel("Distance")
sch.dendrogram(
    complete_linkage,
    leaf_rotation=0.,
    leaf_font_size=8.,
)            
plt.show() 

             
               
#Kmeans clustering
new_df = pd.DataFrame(pca_values[:,0:7])
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3)
kmeans.fit(new_df)
kmeans.labels_

comparing the clusters obtained by n=3 in case of hierarchial and kmeans
from sklearn.decomposition import IncrementalPCA
pca_final = IncrementalPCA(n_components= 3)
X_pca_final = pd.DataFrame(pca_final.fit_transform(X_scaled))
X_pca_final.head()
a=complete_cluster_labels
b= kmeans.labels_
X_pca_final_df = pd.DataFrame(pca_values)
X_pca_final_df["K_means_cluster_ID"]=b
X_pca_final_df["hierarchial cluster labels"]=a
final = X_pca_final_df.head()


