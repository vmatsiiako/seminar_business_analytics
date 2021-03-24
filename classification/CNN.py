import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

#rename these two to df.. and rewrite code
train = pd.read_csv("../Data/sign_mnist_train.csv")
test = pd.read_csv("../Data/sign_mnist_test.csv")

'''
train.iloc[:, 1:] will get the data without the label
train.iloc[:,0] will only get the label
test_size will split the size of the data proportonially to 10% or 0.1
random_state is a constant value while train_test_split, shuffle the dataset
'''
#train.iloc[:, 1:] will get the data without the label
X_train, X_valid, y_train, y_valid = train_test_split(train.iloc[:,1:], train.iloc[:,0], test_size=0.1, random_state=42)
X_test, y_test = test.iloc[:,1:], test.iloc[:,0]


X_train = X_train.values.reshape(-1,28,28,1) / 255.0
X_valid = X_valid.values.reshape(-1,28,28,1) / 255.0
X_test = X_test.values.reshape(-1,28,28,1) / 255.0

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding="SAME"),
    keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding="SAME"),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding="SAME"),
    keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding="SAME"),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=25, activation='softmax'),
])

#get some sort of summary of what we do above
model.summary()

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer = 'nadam',
    metrics=['accuracy']
)
history = model.fit(X_train,y_train,
                   validation_data=(X_valid,y_valid),
                   epochs=10,)

pd.DataFrame(history.history).plot()

model.save("sign_mnist_train.h5")
pred = model.predict(X_test)
pred = np.argmax(pred,axis = 1)
import sklearn
sklearn.metrics.accuracy_score(pred,y_test)