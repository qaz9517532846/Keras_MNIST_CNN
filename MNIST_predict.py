from __future__ import print_function
import keras
import numpy as np
import cv2 as cv
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows = 28
img_cols = 28

img = cv.imread('mnist_test_56.png')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = img.reshape(1, img_rows, img_cols, 1)
img = np.array(img, dtype = np.float32)

model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (img_rows, img_cols, 1)))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation = 'softmax'))

model.load_weights('MNIST_model.h5', by_name = True)
output = model.predict(img) # output predict result
output = output.reshape(-1) # result data 2-d matrix convert 1-d matrix
output = output.tolist() # np-array to list
output = output.index(max(output)) # find max position
print(output)
