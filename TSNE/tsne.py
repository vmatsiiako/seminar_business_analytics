import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cv2
import seaborn as sns

# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html
df = pd.read_csv("../Data/sign_mnist_train.csv")

features = df.columns[1:]
X_train = df.loc[:, features].values
y_train = df.iloc[:,0].values

#load in embedding from autoencoders on train data
df_ae = pd.read_csv("../Data/reduced_trainset_with_BATCH_SIZE_64_NOISE_TYPE_gaussian_NOISE_PERCENTAGE_2_HIDDEN_LAYERS_[620,330,100,13]_LEATNING_RATE_0,002_EPOCH_70.csv", header=None)
X_train_ae = df_ae.iloc[:,0:].values

# increase the contract of pictures
X_contrast = np.zeros(np.shape(X_train))
for i in range(len(X_contrast)):
    image = X_train[i, :]
    image = image.astype(np.uint8)
    X_contrast[i] = cv2.equalizeHist(image).reshape(1, 784)

#run PCA with 13 components
pca = PCA(n_components=13)
princa = pca.fit_transform(X_contrast)

#run tsne on PCA
#TSNE = TSNE(n_components=2, perplexity=5)
#TSNE_output = TSNE.fit_transform(princa)

#run tsne on AE
TSNE = TSNE(n_components=2, perplexity=5)
TSNE_output = TSNE.fit_transform(X_train_ae)
n_iter =TSNE_output.n_iter_
print(n_iter)

#run tsne on full data
#TSNE = TSNE(n_components=2, perplexity=5)
#TSNE_output = TSNE.fit_transform(X_contrast)

tsneDf = pd.DataFrame(data = TSNE_output)
finalDf = pd.concat([tsneDf, df[['label']]], axis=1)

finalDf['tsne1'] = finalDf.loc[:,0].values
finalDf['tsne2'] = finalDf.loc[:,1].values

plt.figure(figsize=(12,8))
plt.title("Tsne with 2 components")
sns.scatterplot(
    x="tsne1", y="tsne2",
    hue = y_train,
    palette=sns.color_palette("hls", 24),
    data=finalDf,
    legend="full"
)
plt.show()