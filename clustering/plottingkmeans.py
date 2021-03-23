import pandas as pd
import matplotlib
from sklearn.metrics import silhouette_score
import seaborn as sns

matplotlib.use('TkAgg')
#https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
#https://towardsdatascience.com/explaining-k-means-clustering-5298dc47bad6

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# load in the data
df = pd.read_csv("../Data/sign_mnist_train.csv")
features = df.columns[1:]
X_train = df.loc[:, features].values
y = df.loc[:,['label']].values

pca = PCA(n_components=5)
#x = df.loc[:, features].values
x = StandardScaler().fit_transform(X_train)
princa = pca.fit_transform(x)

scaler = StandardScaler()
scaler.fit(df)
X_scale = scaler.transform(df)
df_scale = pd.DataFrame(X_scale, columns=df.columns)
df_scale.head()

kmeans = KMeans(init="k-means++", n_clusters=3, n_init=10)
kmeans_pca = kmeans.fit(princa)
print('KMeans PCA Scaled Silhouette Score: {}'.format(silhouette_score(df_scale, kmeans_pca.labels_, metric='euclidean')))

labels_pca_scale = kmeans_pca.labels_
clusters_pca_scale = pd.concat([df_scale, pd.DataFrame({'pca_clusters':labels_pca_scale})], axis=1)

plt.figure(figsize = (10,10))
sns.scatterplot(clusters_pca_scale.iloc[:,0],clusters_pca_scale.iloc[:,1], hue=labels_pca_scale, palette='Set1', s=100, alpha=0.2).set_title('KMeans Clusters (24) Derived from PCA', fontsize=15)
plt.legend()
#plt.show()