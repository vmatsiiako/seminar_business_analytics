import pandas as pd
import matplotlib
import cv2
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# https://www.kaggle.com/saptarsi/kmedoid-sg
# https://towardsdatascience.com/explaining-k-means-clustering-5298dc47bad6

# load in the data
df = pd.read_csv("../Data/sign_mnist_train.csv")
X_train = df.iloc[:,1:].values
y_train = df.iloc[:,0].values

#contrast data
X_contrast = np.zeros(np.shape(X_train))
for i in range(len(X_contrast)):
    image = X_train[i,:]#PCA on scaled data
    image = image.astype(np.uint8)
    X_contrast[i] = cv2.equalizeHist(image).reshape(1,784)
    
X_contrast = X_contrast.astype('float32') / 255.0 - 0.5

scaler = StandardScaler().fit(X_contrast)
X_scale = scaler.transform(X_contrast)
#df_scaled = scaler.transform(df)
#df_scale = pd.DataFrame(X_scale, columns=df.columns)
#df_scale.head()

#scale data
#scaler = StandardScaler().fit(X_contrast)
#scaler = StandardScaler()
#scaler.fit(df)
#X_scale = scaler.transform(df)
#X_scaled = scaler.transform(X_contrast)
#df_scaled = pd.DataFrame(X_scale, columns=df.columns)


#pam of full data
pam_scale = KMedoids(n_clusters = 24, random_state = 0)
pam_full_fit = pam_scale.fit(X_scale)
labels_pam_full = pam_full_fit.labels_
clusters_pam_full = pd.concat([X_scale, pd.DataFrame({'pam_clusters':labels_pam_full})], axis=1)

#pam on pca
#pam = KMedoids(n_clusters = 24, random_state = 0)
#pam_pca = pam.fit(pca_df_scale)
#labels_pca_scale = pam_pca.labels_
#clusters_pca_scale = pd.concat([pca_df_scale, pd.DataFrame({'pca_clusters':labels_pca_scale})], axis=1)

#plot fom full  data
plt.figure(figsize = (10,10))
sns.scatterplot(clusters_pam_full.iloc[:,0],clusters_pam_full.iloc[:,1], hue=labels_pam_full, palette='Set1', s=100, alpha=0.2).set_title('PAM Clusters (24) fom full dataset', fontsize=15)
plt.legend()
plt.show()
