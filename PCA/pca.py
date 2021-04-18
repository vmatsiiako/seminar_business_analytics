import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import cv2
import seaborn as sns

#Initialize constants
MAX_BRIGHTNESS = 255
MEAN = 0.5
NUMBER_OF_PIXELS = 784
PICTURE_DIMENSION = 28
INTRINSIC_DIMENSIONALITY = 13

# load in the data
df = pd.read_csv("../Data/sign_mnist_train.csv")
X_train = df.iloc[:,1:].values
y_train = df.iloc[:,0].values

#contrast train data
X_contrast = np.zeros(np.shape(X_train))
for i in range(len(X_contrast)):
    image = X_train[i,:]
    image = image.astype(np.uint8)
    X_contrast[i] = cv2.equalizeHist(image).reshape(1,NUMBER_OF_PIXELS)

# normalize train data
X_contrast = X_contrast.astype('float32') / MAX_BRIGHTNESS - MEAN
X_train = X_train.astype('float32') / MAX_BRIGHTNESS - MEAN

pca = PCA(n_components=INTRINSIC_DIMENSIONALITY)
principalComponents = pca.fit_transform(X_contrast)
print(pca.explained_variance_ratio_)

# 3-Dimensional Plot
# principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2', 'pc3'])
# finalDf = pd.concat([principalDf, df[['label']]], axis=1)
# fig = plt.figure()
# from mpl_toolkits.mplot3d import Axes3D
# ax = Axes3D(fig)
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_zlabel('Principal Component 3', fontsize = 15)
# ax.set_title('3 component PCA', fontsize = 20)
# targets = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
# import matplotlib.cm as cm
# colors = cm.rainbow(np.linspace(0,1,26))
# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf['label'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'pc1']
#                , finalDf.loc[indicesToKeep, 'pc2']
#                , finalDf.loc[indicesToKeep, 'pc3']
#                , c = color
#                , s = 50)
# ax.legend(targets)
# ax.grid()
# plt.show()

#2-Dimensional Plot
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_contrast)
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])

finalDf = pd.concat([principalDf, df[['label']]], axis=1)

plt.figure(figsize=(18,12))
plt.title("2-component PCA")
sns.scatterplot(
    x='pc1', y='pc2',
    hue = y_train,
    palette=sns.color_palette("hls", 24),
    data=finalDf,
    legend="full"
)
plt.show()

#Print the explained variance
print('Explained variance{}'.format(pca.explained_variance_ratio_))


# THIS PART OF THE CODE IS USED FOR VISUALIZING FEATURES' DISTRIBUTION
# sns.displot(pd.DataFrame(x[:,153]), x=0, binwidth=3)  # If you wanna visualize a distribution of a certain pixel
#sns.displot(X_train, x=0, binwidth=3)   # If you wanna visualize the distribution of a certain picture
#plt.xlim(0, MAX_BRIGHTNESS)
#plt.show()

# This is the code for visualizing the average picture
#plt.figure(figsize=(4, 4))
# Display original
#ax = plt.subplot(1, 1, 1)
#plt.imshow(X_train[0].reshape(PICTURE_DIMENSION, PICTURE_DIMENSION))
#plt.gray()
#ax.get_xaxis().set_visible(False)
#ax.get_yaxis().set_visible(False)
# plt.show()
# plt.savefig('average_picture_contrast.pdf')
