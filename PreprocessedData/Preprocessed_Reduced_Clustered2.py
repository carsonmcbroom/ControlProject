#import pandas
import pandas as pd

#import x and y data
# x_data = pd.read_csv(r'C:\Users\nflfa\Desktop\Git\ControlProject\RawData\RawData.csv')
# y_data = pd.read_csv(r'C:\Users\nflfa\Desktop\Git\ControlProject\RawData\RawData_Label.csv')

# x_data = pd.read_csv("RawData/RawData.csv", sep = "\t", header = None,)
x_data = pd.read_csv('https://raw.githubusercontent.com/carsonmcbroom/ControlProject/main/RawData/RawData.csv')
y_data = pd.read_csv('https://raw.githubusercontent.com/carsonmcbroom/ControlProject/main/RawData/RawData_Label.csv')


print(x_data.head)
print(y_data.head)
#filling empty values with the attribute mean
x_data = x_data.fillna(x_data.mean())
y_data = y_data.fillna(y_data.mean())

print(x_data)
print(y_data)
#set final x and y values
x = x_data.drop(columns = "Index")
y = y_data['label']

print(x)
print(y)

#importing sklearn to transform and split the data
import sklearn

#x data must be scaled with standard scaler
from sklearn.preprocessing import StandardScaler as SS
scale = SS()
x = scale.fit_transform(x)
print(x)

#split training and test data
from sklearn.model_selection import train_test_split as tts
xtrain, xtest, ytrain, ytest = tts(x, y, test_size = 0.2, random_state = 42)


#time to do dimensionality reduction

#Using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca_data = pca.fit_transform(xtrain)

#Using TSNE
from sklearn.manifold import TSNE
tsne_data = TSNE(n_components = 2).fit_transform(xtrain)

#Using SVD
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components = 2)
svd_data = svd.fit_transform(xtrain)

#Using isomap
from sklearn.manifold import Isomap
iso = Isomap(n_components = 2)
iso_data = iso.fit_transform(xtrain)

#Using LLE
from sklearn.manifold import LocallyLinearEmbedding as LLE
lle_data = LLE(n_components = 2).fit_transform(xtrain)


#plotting reduced data vs ytrain data
import matplotlib.pyplot as plt

plt.plot(pca_data, ytrain)

plt.plot(tsne_data, ytrain)

plt.plot(svd_data, ytrain)

plt.plot(iso_data, ytrain)

plt.plot(lle_data, ytrain)


#plotting as scatter plots for clustering

plt.scatter(pca_data[:,0], pca_data[:,1])
plt.show()

plt.scatter(tsne_data[:,0], tsne_data[:,1])
plt.show()

plt.scatter(svd_data[:,0], svd_data[:,1])
plt.show()

plt.scatter(iso_data[:,0], iso_data[:,1])
plt.show()

plt.scatter(lle_data[:,0], lle_data[:,1])
plt.show()


#Using tsne and svd data for clustering

#starting with KMeans on SVD data

from sklearn.cluster import KMeans as KM
kmeans = KM(init = "k-means++", n_clusters = 5, n_init = 10).fit(svd_data)
p_svd = kmeans.predict(svd_data)
plt.scatter(svd_data[:,0], svd_data[:,1], c=p_svd)


#Kmeans on TSNE data

kmeans = KM(init = "k-means++", n_clusters = 5, n_init = 10).fit(tsne_data)
p_tsne = kmeans.predict(tsne_data)
plt.scatter(tsne_data[:,0], tsne_data[:,1], c=p_tsne)


#Now using DBScan (Density Based)
from sklearn.cluster import DBSCAN

#in order to use DBScan, epsilon must be chosen and can be determined from nearest neighbors
from sklearn.neighbors import NearestNeighbors as NN
import numpy as np
nabrs = NN(n_neighbors = 2).fit(svd_data)
distance, index = nabrs.kneighbors(svd_data)
distance = np.sort(distance, axis = 0)
distance = distance[:,1]
plt.grid()
plt.plot(distance)

#0.5 looks like the point of maximum curvature
db = DBSCAN(eps = 0.5).fit(svd_data)
clusters = db.labels_
color_list = ['green', 'blue', 'red', 'yellow',  'orange',  'magenta', 'cyan', 'purple']
vectorizer = np.vectorize(lambda x: color_list[x % len(color_list)])
plt.scatter(svd_data[:,0], svd_data[:,1], c = vectorizer(clusters))

#0.5 looks awful trying 0.75
db = DBSCAN(eps = 0.75).fit(svd_data)
clusters = db.labels_
color_list = ['green', 'blue', 'red', 'yellow',  'orange',  'magenta', 'cyan', 'purple']
vectorizer = np.vectorize(lambda x: color_list[x % len(color_list)])
plt.scatter(svd_data[:,0], svd_data[:,1], c = vectorizer(clusters))

#testing 1
db = DBSCAN(eps = 1).fit(svd_data)
clusters = db.labels_
color_list = ['green', 'blue', 'red', 'yellow',  'orange',  'magenta', 'cyan', 'purple']
vectorizer = np.vectorize(lambda x: color_list[x % len(color_list)])
plt.scatter(svd_data[:,0], svd_data[:,1], c = vectorizer(clusters))


#doing dbscan for TSNE data
nabrs = NN(n_neighbors = 2).fit(tsne_data)
distance, index = nabrs.kneighbors(tsne_data)
distance = np.sort(distance, axis = 0)
distance = distance[:,1]
plt.grid()
plt.plot(distance)

#1.25 looks like a good value for epsilon
db = DBSCAN(eps = 1.25).fit(tsne_data)
clusters = db.labels_
color_list = ['green', 'blue', 'red', 'yellow',  'orange',  'magenta', 'cyan', 'purple']
vectorizer = np.vectorize(lambda x: color_list[x % len(color_list)])
plt.scatter(tsne_data[:,0], tsne_data[:,1], c = vectorizer(clusters))

#trying 1.5
db = DBSCAN(eps = 1.5).fit(tsne_data)
clusters = db.labels_
color_list = ['green', 'blue', 'red', 'yellow',  'orange',  'magenta', 'cyan', 'purple']
vectorizer = np.vectorize(lambda x: color_list[x % len(color_list)])
plt.scatter(tsne_data[:,0], tsne_data[:,1], c = vectorizer(clusters))

#trying 1
db = DBSCAN(eps = 1).fit(tsne_data)
clusters = db.labels_
color_list = ['green', 'blue', 'red', 'yellow',  'orange',  'magenta', 'cyan', 'purple']
vectorizer = np.vectorize(lambda x: color_list[x % len(color_list)])
plt.scatter(tsne_data[:,0], tsne_data[:,1], c = vectorizer(clusters))

#trying 2
db = DBSCAN(eps = 2).fit(tsne_data)
clusters = db.labels_
color_list = ['green', 'blue', 'red', 'yellow',  'orange',  'magenta', 'cyan', 'purple']
vectorizer = np.vectorize(lambda x: color_list[x % len(color_list)])
plt.scatter(tsne_data[:,0], tsne_data[:,1], c = vectorizer(clusters))

#trying 3
db = DBSCAN(eps = 3).fit(tsne_data)
clusters = db.labels_
color_list = ['green', 'blue', 'red', 'yellow',  'orange',  'magenta', 'cyan', 'purple']
vectorizer = np.vectorize(lambda x: color_list[x % len(color_list)])
plt.scatter(tsne_data[:,0], tsne_data[:,1], c = vectorizer(clusters))


#Final clustering method will be Agglomerative Clustering (AC)
from sklearn.cluster import AgglomerativeClustering as AC
import scipy.cluster.hierarchy as sch

#The highest vertical distance that doesn't intersect with any clusters gives the optimal number of clusters which would be 
# 2 in this case. But we know that there are 5 individual labels so we'll use 5. 

dendrogram = sch.dendrogram(sch.linkage(svd_data, method='ward'))

#Using agglomerative clustering with 5 clusters
svd_clusters = AC(n_clusters=5).fit(svd_data)
color = svd_clusters.labels_
plt.scatter(svd_data[:,0], svd_data[:,1], c = color)

#Now doing AC with tsne data

#Looks like 3 might be the best, but, again, with knowledge of the labels we'll use 5. 
dendrogram = sch.dendrogram(sch.linkage(tsne_data, method='ward'))

tsne_clusters = AC(n_clusters=5).fit(tsne_data)
color = tsne_clusters.labels_
plt.scatter(tsne_data[:,0], tsne_data[:,1], c = color)