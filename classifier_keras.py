from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from helper import *
import os

# tf.python.control_flow_ops = tf
# from keras.layers.core import Activation, Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPool2D
# from keras.models import Sequential
from helper import save_data
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


""" create a pickle file """
import numpy as np
import cv2 as cv
img1, img2 = cv.imread("PET-bottle.png"), cv.imread("Not-PET-bottle.png")
features_ = np.array([img1, img2])
labels_ = np.array([1, 2])
save_data("pet-bottles.p", features_, labels_)
f, l = load_data("pet-bottles.p")
cv.imshow("img: ", f[0])
cv.waitKey()
print("label: ", labels_[0])

"""
x_train, y_train = load_data('transforms-train.p')
x_test, y_test = load_data('test.p')

model = Sequential()
# Layer 1 -- convolution
model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='valid', input_shape=(32, 32, 3),
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
# Layer 2 -- convolution    
model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
# Layer 3 -- convolution
model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu'))
# flatten
model.add(Flatten(input_shape=(32, 32, 3)))
# Layer 4 -- fully connected layer
model.add(Dense(units=120))
model.add(Dropout(rate=0.5))
model.add(Activation(activation='relu'))
# Layer 5 -- fully connected layer
model.add(Dense(units=84))
model.add(Activation(activation='relu'))
model.add(Dropout(rate=0.5))
# Layer 6 -- output layer
model.add(Dense(units=43))
model.add(Activation(activation='softmax'))
# training
x_train, y_train = pre_process(x_train, y_train)
binarizer = LabelBinarizer()
y_train_one_hot = binarizer.fit_transform(y_train)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train, y=y_train_one_hot, batch_size=128,
          epochs=3, validation_split=0.3, shuffle=True, verbose=0)
# testing
x_test, y_test = pre_process(x_test, y_test)
binarizer = LabelBinarizer()
y_test_one_hot = binarizer.fit_transform(y_test)
metrics = model.evaluate(x=x_test, y=y_test_one_hot, verbose=0)
for metrics_index, metrics_name in enumerate(model.metrics_names):
    name = metrics_name
    value = metrics[metrics_index]
    print("{}: {}".format(name, value))
"""