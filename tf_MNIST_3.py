from __future__ import print_function
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers

batch_size = 128
num_classes = 10
epochs = 10

# input image dimensions
img_rows = 28
img_cols = 28

# the data, split between train and test sets
train_data = np.load('train_data.npy') 
test_data = np.load('test_data.npy') 

train_x = [i[0] for i in train_data]
print(np.shape(train_x))

train_y = [i[1] for i in train_data]
print(np.shape(train_y))

test_x = [i[0] for i in test_data]
print(np.shape(test_x))

test_y = [i[1] for i in test_data]
print(np.shape(test_y))

train_x = np.array(train_x, dtype = np.float32)
train_y = np.array(train_y, dtype = np.float32)
test_x = np.array(test_x, dtype = np.float32)
test_y = np.array(test_y, dtype = np.float32)

model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (img_rows, img_cols, 1)))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation = 'softmax'))

sgd = optimizers.SGD(lr = 0.0001, decay = 1e-6, momentum = 0.9, nesterov = True)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer = sgd,
              metrics=['accuracy'])

result = model.fit(train_x, train_y, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (train_x, train_y))

score = model.evaluate(test_x, test_y, verbose = 0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# save model
model.save_weights('MNIST_model.h5')

# Plot training & validation accuracy values
plt.plot(result.history['accuracy'])
plt.plot(result.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()