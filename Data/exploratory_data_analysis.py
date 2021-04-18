import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import seaborn as sns
from constants import MAX_BRIGHTNESS, PICTURE_DIMENSION, NUMBER_OF_PIXELS, MEAN


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

# THIS PART OF THE CODE IS USED FOR VISUALIZING FEATURES' DISTRIBUTION
sns.displot(pd.DataFrame(X_train[:,153]), x=0, binwidth=3)  # If you wanna visualize a distribution of a certain pixel
sns.displot(X_train, x=0, binwidth=3)   # If you wanna visualize the distribution of a certain picture
plt.xlim(0, MAX_BRIGHTNESS)
plt.show()

# This is the code for visualizing the average picture
plt.figure(figsize=(4, 4))
# Display original
ax = plt.subplot(1, 1, 1)
plt.imshow(X_train[0].reshape(PICTURE_DIMENSION, PICTURE_DIMENSION))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
plt.savefig('average_picture_contrast.pdf')
