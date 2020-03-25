import numpy as np
import cv2 as cv

fp_train = open("D:/tensorflow_example/tf_MNIST_3/tf_MNIST_3/MNIST_data/mnist_train/train.txt", "r")
fp_test = open("D:/tensorflow_example/tf_MNIST_3/tf_MNIST_3/MNIST_data/mnist_test/test.txt", "r")
line_train = fp_train.readline()
line_test = fp_test.readline()

training_data = []
test_data = []

def lable(img_label):
    if   img_label == '0': return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif img_label == '1': return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif img_label == '2': return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif img_label == '3': return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif img_label == '4': return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif img_label == '5': return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif img_label == '6': return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif img_label == '7': return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif img_label == '8': return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif img_label == '9': return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

## 用 while 逐行讀取檔案內容，直至檔案結尾
while line_train:
    x = line_train.split( )
    print('Processing_image: ' + x[0])
    img = cv.imread("D:/tensorflow_example/tf_MNIST_3/tf_MNIST_3/MNIST_data/mnist_train/" + x[1] + "/" + x[0], cv.IMREAD_GRAYSCALE)  
    img = cv.resize(img, (28, 28))
    img = img.reshape(28, 28, 1)
    training_data.append([np.array(img, dtype="float32"), np.array(lable(x[1]), dtype="float32")])
    line_train = fp_train.readline()

fp_train.close()
np.save('train_data.npy', training_data) 
print("train data processing finished.")

while line_test:
    x = line_test.split( )
    print('Processing_image: ' + x[0])
    img = cv.imread("D:/tensorflow_example/tf_MNIST_3/tf_MNIST_3/MNIST_data/mnist_test/" + x[1] + "/" + x[0], cv.IMREAD_GRAYSCALE)  
    img = cv.resize(img, (28, 28))
    img = img.reshape(28, 28, 1)
    test_data.append([np.array(img, dtype="float32"), np.array(lable(x[1]), dtype="float32")])
    line_test = fp_test.readline()

fp_test.close()
np.save('test_data.npy', test_data) 
print("test data processing finished.")
