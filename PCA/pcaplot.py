import pandas as pd
import matplotlib
matplotlib.use('TkAgg')

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("/Users/anoukveltman/Downloads/archive/sign_mnist_train.csv")

features = df.columns[1:]
x = df.loc[:, features].values
y = df.loc[:,['label']].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=5)
principalComponents = pca.fit_transform(x)
#principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2', 'pc3', 'pca4', 'pca5', 'pca6'])
approximation = pca.inverse_transform(principalComponents)

plt.figure(figsize=(8,4));

# Original Image
plt.subplot(1, 2, 1);
plt.imshow(x[3].reshape(28,28),
           cmap = plt.cm.gray);
plt.xlabel('784 components', fontsize = 14)
plt.title('Original Image', fontsize = 20)

# 154 principal components
plt.subplot(1, 2, 2);
plt.imshow(approximation[3].reshape(28, 28),
              cmap = plt.cm.gray);
plt.xlabel('5 components', fontsize = 14)
plt.title('Reconstructed Image', fontsize = 20)
plt.show()