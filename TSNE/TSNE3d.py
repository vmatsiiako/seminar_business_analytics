import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import cv2
import seaborn as sns

# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html
df = pd.read_csv("../Data/sign_mnist_train.csv")

features = df.columns[1:]
X_train = df.loc[:, features].values
y_train = df.iloc[:,0].values

# increase the contract of pictures
X_contrast = np.zeros(np.shape(X_train))
for i in range(len(X_contrast)):
    image = X_train[i, :]
    image = image.astype(np.uint8)
    X_contrast[i] = cv2.equalizeHist(image).reshape(1, 784)

pca = PCA(n_components=13)
# x = df.loc[:, features].values
x = StandardScaler().fit_transform(X_contrast)
princa = pca.fit_transform(x)

#run tsne
TSNE = TSNE(n_components=3)
TSNE_output = TSNE.fit_transform(princa)

tsneDf = pd.DataFrame(data = TSNE_output, columns = ['tsne1', 'tsne2', 'tsne3'])
finalDf = pd.concat([tsneDf, df[['label']]], axis=1)

fig = plt.figure()
from mpl_toolkits.mplot3d import Axes3D
ax = Axes3D(fig)
ax.set_xlabel('TSNE 1', fontsize = 15)
ax.set_ylabel('TSNE 2', fontsize = 15)
ax.set_zlabel('TSNE 3', fontsize = 15)
ax.set_title('3 component TSNE', fontsize = 20)
targets = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0,1,26))
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['label'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'tsne1']
               , finalDf.loc[indicesToKeep, 'tsne2']
               , finalDf.loc[indicesToKeep, 'tsne3']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()