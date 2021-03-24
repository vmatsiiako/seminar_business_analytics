import pandas
import pandas as pd
import matplotlib
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, v_measure_score
import cv2
matplotlib.use('TkAgg')
#https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
#https://towardsdatascience.com/explaining-k-means-clustering-5298dc47bad6
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("../Data/sign_mnist_train.csv")
X_train = df.iloc[:,1:].values
y_train = df.iloc[:,0].values

#contrast data
X_contrast = np.zeros(np.shape(X_train))
for i in range(len(X_contrast)):
    image = X_train[i,:]
    image = image.astype(np.uint8)
    X_contrast[i] = cv2.equalizeHist(image).reshape(1,784)

# normalize data
X_contrast = X_contrast.astype('float32') / 255.0 - 0.5
X_train = X_train.astype('float32') / 255.0 - 0.5

#run PCA with n=13 principal components
pca = PCA(n_components=13)
princa = pca.fit_transform(X_contrast)

kmeans = KMeans(init="k-means++", n_clusters=24, n_init=4)
#kmeans_full = kmeans.fit(X_contrast)
#labels = kmeans.predict(X_contrast)
kmeans_pca = kmeans.fit(princa)
labels_pca = kmeans.predict(princa)

# homo = homogeneity_score(y_train, labels)
# print(homo)
# comp = completeness_score(y_train, labels)
# print(comp)
# v = v_measure_score(y_train, labels)
# print(v)

homo_pca = homogeneity_score(y_train, labels_pca)
comp_pca =completeness_score(y_train, labels_pca)
print(homo_pca)
print(comp_pca)

# cluslabs = list(zip(y_train,labels))
# #print(cluslabs)
# cluslabdf = pandas.DataFrame(data=cluslabs)
# #cluslabdf.to_csv("cluslabs.csv", sep=',',index=False)

#kmeans_pca = kmeans.fit(princa)
#print('KMeans PCA Scaled Silhouette Score: {}'.format(silhouette_score(df_scale, kmeans_pca.labels_, metric='euclidean')))

#kmeans_full = kmeans.fit(x)
#print('KMeans full Scaled Silhouette Score: {}'.format(silhouette_score(df_scale, kmeans_full.labels_, metric='euclidean')))

#h = .02

# Plot the decision boundary. For that, we will assign a color to each
#x_min, x_max = princa[:, 0].min() - 1, princa[:, 0].max() + 1
#y_min, y_max = princa[:, 1].min() - 1, princa[:, 1].max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
#Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
#Z = Z.reshape(xx.shape)
# plt.figure(1)
# plt.clf()
# plt.imshow(Z, interpolation="nearest",
#            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#            cmap=plt.cm.Paired, aspect="auto", origin="lower")

# plt.plot(princa[:, 0], princa[:, 1], 'k.', markersize=2)
# # Plot the centroids as a white X
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3,
#             color="w", zorder=10)
# plt.title("K-means clustering on the digits dataset (PCA-reduced data)\n"
#           "Centroids are marked with white cross")
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()

#N_clusters = 25

#kmeans = KMeans(init="k-means++", n_clusters= N_clusters, n_init=4, random_state=0)
#kmeans.fit(X_train)

# Run the Kmeans algorithm and get the index of data points clusters
#sse = []
#list_k = list(range(1, 10))

#for k in list_k:
#    km = KMeans(n_clusters=k)
#    km.fit(X_train)
#    sse.append(km.inertia_)

# Plot sse against k
#plt.figure(figsize=(6, 6))
#plt.plot(list_k, sse, '-o')
#plt.xlabel(r'Number of clusters *k*')
#plt.ylabel('Sum of squared distance');
#plt.show()
#print(sse)