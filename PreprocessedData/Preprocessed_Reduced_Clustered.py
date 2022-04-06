#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import pandas
import pandas as pd

#import x and y data
x_data = pd.read_csv(r'https://raw.githubusercontent.com/carsonmcbroom/ControlProject/main/RawData/RawData.csv')
y_data = pd.read_csv(r'https://raw.githubusercontent.com/carsonmcbroom/ControlProject/main/RawData/RawData_Label.csv')

print(x_data)
print(y_data)


# In[2]:


#filling empty values with the attribute mean
x_data = x_data.fillna(x_data.mean())
y_data = y_data.fillna(y_data.mean())

print(x_data)
print(y_data)


# In[3]:


#set final x and y values
x = x_data.drop(columns = "Index")
y = y_data['label']

print(x)
print(y)


# In[4]:


#importing sklearn to transform and split the data
import sklearn

#x data must be scaled with standard scaler
from sklearn.preprocessing import StandardScaler as SS
scale = SS()
x = scale.fit_transform(x)
print(x)


# In[5]:


#split training and test data
from sklearn.model_selection import train_test_split as tts
xtrain, xtest, ytrain, ytest = tts(x, y, test_size = 0.2, random_state = 42)


# In[6]:


#time to do dimensionality reduction
#Using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca_data = pca.fit_transform(xtrain)


# In[7]:


#Using TSNE
from sklearn.manifold import TSNE
tsne_data = TSNE(n_components = 2).fit_transform(xtrain)


# In[8]:


#Using SVD
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components = 2)
svd_data = svd.fit_transform(xtrain)


# In[9]:


#Using isomap
from sklearn.manifold import Isomap
iso = Isomap(n_components = 2)
iso_data = iso.fit_transform(xtrain)


# In[10]:


#Using LLE
from sklearn.manifold import LocallyLinearEmbedding as LLE
lle_data = LLE(n_components = 2).fit_transform(xtrain)


# In[11]:


#plotting reduced data vs ytrain data
import matplotlib.pyplot as plt
plt.plot(pca_data, ytrain)


# In[12]:


plt.plot(tsne_data, ytrain)


# In[13]:


plt.plot(svd_data, ytrain)


# In[14]:


plt.plot(iso_data, ytrain)


# In[15]:


plt.plot(lle_data, ytrain)


# In[16]:


#plotting as scatter plots for clustering
plt.scatter(pca_data[:,0], pca_data[:,1])
plt.show()


# In[17]:


plt.scatter(tsne_data[:,0], tsne_data[:,1])
plt.show()


# In[18]:


plt.scatter(svd_data[:,0], svd_data[:,1])
plt.show()


# In[19]:


plt.scatter(iso_data[:,0], iso_data[:,1])
plt.show()


# In[20]:


plt.scatter(lle_data[:,0], lle_data[:,1])
plt.show()


# In[25]:


#Using tsne and svd data for clustering
#starting with KMeans
from sklearn.cluster import KMeans as KM
kmeans = KM(init = "k-means++", n_clusters = 2, n_init = 10).fit(svd_data)
p_svd = kmeans.predict(svd_data)
plt.scatter(svd_data[:,0], svd_data[:,1], c=p_svd)


# In[26]:


#Testing different number of clusters
kmeans = KM(init = "k-means++", n_clusters = 3, n_init = 10).fit(svd_data)
p_svd = kmeans.predict(svd_data)
plt.scatter(svd_data[:,0], svd_data[:,1], c=p_svd)


# In[28]:


kmeans = KM(init = "k-means++", n_clusters = 4, n_init = 10).fit(svd_data)
p_svd = kmeans.predict(svd_data)
plt.scatter(svd_data[:,0], svd_data[:,1], c=p_svd)


# In[32]:


#Kmeans on tsne data
kmeans = KM(init = "k-means++", n_clusters = 2, n_init = 10).fit(tsne_data)
p_tsne = kmeans.predict(tsne_data)
plt.scatter(tsne_data[:,0], tsne_data[:,1], c=p_tsne)


# In[35]:


#Testing different cluster sizes
kmeans = KM(init = "k-means++", n_clusters = 3, n_init = 10).fit(tsne_data)
p_tsne = kmeans.predict(tsne_data)
plt.scatter(tsne_data[:,0], tsne_data[:,1], c=p_tsne)


# In[33]:


kmeans = KM(init = "k-means++", n_clusters = 4, n_init = 10).fit(tsne_data)
p_tsne = kmeans.predict(tsne_data)
plt.scatter(tsne_data[:,0], tsne_data[:,1], c=p_tsne)


# In[34]:


kmeans = KM(init = "k-means++", n_clusters = 10, n_init = 10).fit(tsne_data)
p_tsne = kmeans.predict(tsne_data)
plt.scatter(tsne_data[:,0], tsne_data[:,1], c=p_tsne)


# In[36]:


#Now using DBScan
from sklearn.cluster import DBSCAN


# In[39]:


#in order to use DBScan epsilon must be chosen and can be determined from nearest neighbors
from sklearn.neighbors import NearestNeighbors as NN
import numpy as np
nabrs = NN(n_neighbors = 2).fit(svd_data)
distance, index = nabrs.kneighbors(svd_data)
distance = np.sort(distance, axis = 0)
distance = distance[:,1]
plt.grid()
plt.plot(distance)


# In[48]:


#0.5 looks like the point of maximum curvature
db = DBSCAN(eps = 0.5).fit(svd_data)
clusters = db.labels_
color_list = ['green', 'blue', 'red', 'yellow',  'orange',  'magenta', 'cyan', 'purple']
vectorizer = np.vectorize(lambda x: color_list[x % len(color_list)])
plt.scatter(svd_data[:,0], svd_data[:,1], c = vectorizer(clusters))


# In[50]:


#0.5 looks awful trying 0.75
db = DBSCAN(eps = 0.75).fit(svd_data)
clusters = db.labels_
color_list = ['green', 'blue', 'red', 'yellow',  'orange',  'magenta', 'cyan', 'purple']
vectorizer = np.vectorize(lambda x: color_list[x % len(color_list)])
plt.scatter(svd_data[:,0], svd_data[:,1], c = vectorizer(clusters))


# In[51]:


#testing 1
db = DBSCAN(eps = 1).fit(svd_data)
clusters = db.labels_
color_list = ['green', 'blue', 'red', 'yellow',  'orange',  'magenta', 'cyan', 'purple']
vectorizer = np.vectorize(lambda x: color_list[x % len(color_list)])
plt.scatter(svd_data[:,0], svd_data[:,1], c = vectorizer(clusters))


# In[52]:


#doing dbscan for tsne data
nabrs = NN(n_neighbors = 2).fit(tsne_data)
distance, index = nabrs.kneighbors(tsne_data)
distance = np.sort(distance, axis = 0)
distance = distance[:,1]
plt.grid()
plt.plot(distance)


# In[53]:


#1.25 looks like a good value for epsilon
db = DBSCAN(eps = 1.25).fit(tsne_data)
clusters = db.labels_
color_list = ['green', 'blue', 'red', 'yellow',  'orange',  'magenta', 'cyan', 'purple']
vectorizer = np.vectorize(lambda x: color_list[x % len(color_list)])
plt.scatter(tsne_data[:,0], tsne_data[:,1], c = vectorizer(clusters))


# In[54]:


#trying 1.5
db = DBSCAN(eps = 1.5).fit(tsne_data)
clusters = db.labels_
color_list = ['green', 'blue', 'red', 'yellow',  'orange',  'magenta', 'cyan', 'purple']
vectorizer = np.vectorize(lambda x: color_list[x % len(color_list)])
plt.scatter(tsne_data[:,0], tsne_data[:,1], c = vectorizer(clusters))


# In[55]:


#trying 1
db = DBSCAN(eps = 1).fit(tsne_data)
clusters = db.labels_
color_list = ['green', 'blue', 'red', 'yellow',  'orange',  'magenta', 'cyan', 'purple']
vectorizer = np.vectorize(lambda x: color_list[x % len(color_list)])
plt.scatter(tsne_data[:,0], tsne_data[:,1], c = vectorizer(clusters))


# In[56]:


#trying 2
db = DBSCAN(eps = 2).fit(tsne_data)
clusters = db.labels_
color_list = ['green', 'blue', 'red', 'yellow',  'orange',  'magenta', 'cyan', 'purple']
vectorizer = np.vectorize(lambda x: color_list[x % len(color_list)])
plt.scatter(tsne_data[:,0], tsne_data[:,1], c = vectorizer(clusters))


# In[57]:


#trying 3
db = DBSCAN(eps = 3).fit(tsne_data)
clusters = db.labels_
color_list = ['green', 'blue', 'red', 'yellow',  'orange',  'magenta', 'cyan', 'purple']
vectorizer = np.vectorize(lambda x: color_list[x % len(color_list)])
plt.scatter(tsne_data[:,0], tsne_data[:,1], c = vectorizer(clusters))


# In[61]:


#Final clustering method will be Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering as AC
import scipy.cluster.hierarchy as sch
#The highest vertical distance that doesn't intersect with any clusters gives the optimal number of clusters
#It looks like 2 is the optimal number
dendrogram = sch.dendrogram(sch.linkage(svd_data, method='ward'))


# In[63]:


#Using agglomerative clustering with 2 clusters
svd_clusters = AC(n_clusters=2).fit(svd_data)
color = svd_clusters.labels_
plt.scatter(svd_data[:,0], svd_data[:,1], c = color)


# In[64]:


#trying with 3
svd_clusters = AC(n_clusters=3).fit(svd_data)
color = svd_clusters.labels_
plt.scatter(svd_data[:,0], svd_data[:,1], c = color)


# In[65]:


#With 4
svd_clusters = AC(n_clusters=4).fit(svd_data)
color = svd_clusters.labels_
plt.scatter(svd_data[:,0], svd_data[:,1], c = color)


# In[66]:


#Now doing AC with tsne data
#Looks like 3 might be the best
dendrogram = sch.dendrogram(sch.linkage(tsne_data, method='ward'))


# In[68]:


tsne_clusters = AC(n_clusters=3).fit(tsne_data)
color = tsne_clusters.labels_
plt.scatter(tsne_data[:,0], tsne_data[:,1], c = color)


# In[73]:


#trying with 5
tsne_clusters = AC(n_clusters=5).fit(tsne_data)
color = tsne_clusters.labels_
plt.scatter(tsne_data[:,0], tsne_data[:,1], c = color)


# In[70]:


#trying with 8
tsne_clusters = AC(n_clusters=8).fit(tsne_data)
color = tsne_clusters.labels_
plt.scatter(tsne_data[:,0], tsne_data[:,1], c = color)


# In[74]:


#with 4
tsne_clusters = AC(n_clusters=4).fit(tsne_data)
color = tsne_clusters.labels_
plt.scatter(tsne_data[:,0], tsne_data[:,1], c = color)

