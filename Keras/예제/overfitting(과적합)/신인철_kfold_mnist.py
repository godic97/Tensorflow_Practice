from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from sklearn.model_selection import KFold
import numpy as np

(x_train, y_train), (x_val, y_val) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 784).astype("float32") / 255
x_val = x_val.reshape(x_val.shape[0], 784).astype("float32") /  255

from keras.utils import to_categorical
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)

X = np.concatenate((x_train, x_val), axis=0)
Y = np.concatenate((y_train, y_val), axis=0)

kfold = KFold(n_splits=5, shuffle=True)

for train, val in kfold.split(X, Y):
    model = Sequential()

    model.add(Dense(512, input_dim=784, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss="categorical_crossentropy",optimizer="adam", metrics=['accuracy'])
    history = model.fit(X[train], Y[train], batch_size=512, epochs=4, validation_data=(X[val], Y[val]))


import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss) + 1)

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training Loss vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()