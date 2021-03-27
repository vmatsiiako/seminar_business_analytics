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

pca = PCA(n_components=50)
# x = df.loc[:, features].values
x = StandardScaler().fit_transform(X_contrast)
princa = pca.fit_transform(x)

#run tsne on PCA
#TSNE = TSNE(n_components=2, perplexity=40)
#TSNE_output = TSNE.fit_transform(princa)

#run tsne on full data
TSNE = TSNE(n_components=2, perplexity=40)
TSNE_output = TSNE.fit_transform(X_contrast)

#run tsne with internal PCA function
#TSNE = TSNE(n_components=2, perplexity=40, init='pca, n_components=13')
#TSNE_output = TSNE.fit_transform(X_contrast)


tsneDf = pd.DataFrame(data = TSNE_output)
finalDf = pd.concat([tsneDf, df[['label']]], axis=1)

finalDf['tsne-2d-one'] = finalDf.loc[:,0].values
finalDf['tsne-2d-two'] = finalDf.loc[:,1].values

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue = y_train,
    palette=sns.color_palette("hls", 24),
    data=finalDf,
    legend="full"
)
plt.show()