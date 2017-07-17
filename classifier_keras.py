import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from helper import *
import os

tf.python.control_flow_ops = tf
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x_train, y_train = load_data('transforms-train.p')
x_test, y_test = load_data('test.p')

model = Sequential()
# Layer 1 -- convolution
model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='valid', input_shape=(32, 32, 3), activation='relu'))
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

x_train, y_train = pre_process(x_train, y_train)
binarizer = LabelBinarizer()
y_train_one_hot = binarizer.fit_transform(y_train)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# training
model.fit(x=x_train, y=y_train_one_hot, batch_size=128,
          epochs=3, validation_split=0.3, shuffle=True, verbose=0)

x_test, y_test = pre_process(x_test, y_test)
binarizer = LabelBinarizer()
y_test_one_hot = binarizer.fit_transform(y_test)
# testing
metrics = model.evaluate(x=x_test, y=y_test_one_hot, verbose=0)
for metrics_index, metrics_name in enumerate(model.metrics_names):
    name = metrics_name
    value = metrics[metrics_index]
    print("{}: {}".format(name, value))
