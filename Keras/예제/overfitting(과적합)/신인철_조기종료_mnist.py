from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.callbacks import EarlyStopping
(x_train, y_train), (x_val, y_val) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 784).astype("float32") / 255
x_val = x_val.reshape(x_val.shape[0], 784).astype("float32") /  255

from keras.utils import to_categorical
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)

cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10)

model = Sequential()

model.add(Dense(512, input_dim=784, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation="relu"))
model.add(Dense(10, activation='softmax'))

model.compile(loss="categorical_crossentropy",optimizer="adam", metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=512, epochs=10, callbacks=[cb_early_stopping])

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