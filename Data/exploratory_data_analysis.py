# import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from constants import MAX_BRIGHTNESS, PICTURE_DIMENSION
# matplotlib.use('TkAgg')

# load in the data
df = pd.read_csv("../Data/sign_mnist_train.csv")
X_train = df.iloc[:, 1:].values
y_train = df.iloc[:, 0].values
X_train_contrast = df.iloc[:, 1:].values
y_train_contrast = df.iloc[:, 0].values

# Visualize feature distribution for original training data
sns.displot(data=pd.DataFrame(X_train[:, 153]), x=0, binwidth=3)  # visualize a distribution of a certain pixel
sns.displot(data=X_train, x=0, binwidth=3)   # visualize the distribution of a certain picture
plt.xlim(0, MAX_BRIGHTNESS)
plt.show()

# Visualize feature distribution for contrasted training data
sns.displot(pd.DataFrame(X_train_contrast[:, 153]), x=0, binwidth=3)  # visualize a distribution of a certain pixel
sns.displot(X_train_contrast, x=0, binwidth=3)   # visualize the distribution of a certain picture
plt.xlim(0, MAX_BRIGHTNESS)
plt.show()

# This is the code for visualizing the average picture for original training data
plt.figure(figsize=(4, 4))
# Display original
ax = plt.subplot(1, 1, 1)
plt.imshow(X_train[0].reshape(PICTURE_DIMENSION, PICTURE_DIMENSION))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
plt.savefig('average_picture_original.pdf')

# This is the code for visualizing the average picture for contrasted training data
plt.figure(figsize=(4, 4))
# Display original
ax = plt.subplot(1, 1, 1)
plt.imshow(X_train_contrast[0].reshape(PICTURE_DIMENSION, PICTURE_DIMENSION))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
plt.savefig('average_picture_contrast.pdf')
