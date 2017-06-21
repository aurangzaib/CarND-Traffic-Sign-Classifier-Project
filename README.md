**Traffic Sign Recognition**
============================

 

*For the complete implementation and notebook:*

<https://github.com/aurangzaib/CarND-Traffic-Sign-Classifier-Project>

 

**Build a Traffic Sign Recognition Project:**

The goals / steps of this project are the following:

-   Load the German Traffic Signs pre-labelled dataset

-   Explore, summarize and visualize the data set

-   Design, train and test a model architecture with high accuracy

-   Use the model to make predictions on new images from the internet

-   Analyze the softmax probabilities of the new images

 

### **Data Set Summary & Exploration:**

First we will explore together the dataset of traffic signs. We will see what is
a shape of an image, how many training, validation and testing examples are
available in the dataset.

As we will see, the dataset doesn't have a uniform distribution of the samples
for each class.

```python
def get_data_summary(feature, label):
    import numpy as np
    # What's the shape of an traffic sign image?
    image_shape = feature[0].shape
    # How many unique classes/labels there are in the dataset.
    unique_classes, n_samples = np.unique(label,
                                          return_index=False,
                                          return_inverse=False,
                                          return_counts=True)
    n_classes = len(unique_classes)
    n_samples = n_samples.tolist()
    print("Image data shape =", image_shape)
    return image_shape[0], image_shape[2], n_classes, n_samples


def train_test_examples(x_train, x_validation, x_test):
    # Number of training examples
    n_train = len(x_train)
    # Number of validation examples
    n_validation = len(x_validation)
    # Number of testing examples.
    n_test = len(x_test)
    print("Number of training examples =", n_train)
    print("Number of validation examples =", n_validation)
    print("Number of testing examples =", n_test)
```

| **Property**       | **Summary** |
|--------------------|-------------|
| Image shape        | 32x32x3     |
| Training samples   | 34799       |
| Validation samples | 12630       |
| Testing samples    | 4410        |
| Unique classes     | 43          |

 

#### **2. Include an exploratory visualization of the dataset:**

Now we will visualize the dataset, what are the features available and how the
labels are distributed in the dataset:

```python
def get_classes_samples(index, labels):
    return [i for i, _x_ in enumerate(labels) if _x_ == index]


def loopover_data(index, x, y, high_range, steps):
    import matplotlib.pyplot as plt
    % matplotlib inline
    images = get_classes_samples(index, y)
    _images_ = images[:high_range:steps] if len(images) > 100 else images
    imgaes_in_row = int(high_range/steps)
    fig, axes = plt.subplots(1, imgaes_in_row, figsize=(15, 15))
    for _index, image_index in enumerate(_images_):
        image = x[image_index].squeeze()
        axes[_index].imshow(image)
    plt.show()


def visualize_data(x, y, n_classes, n_samples, high_range=160, steps=20, show_desc=True, single_class=False):
    from pandas.io.parsers import read_csv
    label_signs = read_csv('signnames.csv').values[:, 1]  # fetch only sign names
    if single_class:
        loopover_data(n_classes, x, y, high_range, steps)
    else:
        for index in range(n_classes):
            if show_desc:
                print("Class {} -- {} -- {} samples".format(index + 1, label_signs[index], n_samples[index]))
            loopover_data(index, x, y, high_range, steps)
``` 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class 1 -- Speed limit (20km/h) -- 180 samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

![](documentation/output_12_1.png)

 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class 2 -- Speed limit (30km/h) -- 1980 samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

![](documentation/output_12_3.png)

 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class 3 -- Speed limit (50km/h) -- 2010 samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

![](documentation/output_12_5.png)

 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class 4 -- Speed limit (60km/h) -- 1260 samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

![](documentation/output_12_7.png)

 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class 5 -- Speed limit (70km/h) -- 1770 samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

![](documentation/output_12_9.png)

 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class 6 -- Speed limit (80km/h) -- 1650 samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

![](documentation/output_12_11.png)

 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class 7 -- End of speed limit (80km/h) -- 360 samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

![](documentation/output_12_13.png)

 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class 8 -- Speed limit (100km/h) -- 1290 samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

![](documentation/output_12_15.png)

 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class 9 -- Speed limit (120km/h) -- 1260 samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

![](documentation/output_12_17.png)

 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class 10 -- No passing -- 1320 samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

![](documentation/output_12_19.png)

 
```python
def histogram_data(x, n_samples, n_classes):
    import matplotlib.pyplot as plt
    width = 1 / 1.2
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(111)
    ax.set_title('Samples Distribution')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Number of Samples')
    plt.bar(range(n_classes), n_samples, width, color="blue")
    plt.show()
```
 

#### **Labels distribution in Train Dataset:**

![](documentation/output_15_0.png)

 

#### **Labels distribution in Augmented Dataset:**

As we discussed, the dataset contains very few samples for some the classes.
Obviously, we need to fix it.

We will see later how we can fix this issue by augmenting the given dataset but
for now let's enjoy the histogram after the data augmentation.

![](documentation/output_17_1.png)

 

#### **Labels distribution in Test Dataset:**

![](documentation/output_19_1.png)

 

### **Pre-process the Data Set:**

Preprocessing is an important step before training neural network. It consists
of:

-   Grayscale the images.

-   Normalize the dataset using Feature Scaling.

 

[Yann LeCun Paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)
describes that the color channel info doesn't play any useful part in
classification, so we apply grayscaling on the images to have uniform values in
all 3 channels. The images are transformed to 3 channel grayscale using OpenCV.

The train, validation and test datasets are normalized using Feature Rescaling.

```python
def grayscale(x):
    import cv2 as cv
    import numpy as np
    for index, image in enumerate(x):
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        im2 = np.zeros_like(image)
        im2[:, :, 0], im2[:, :, 1], im2[:, :, 2] = gray, gray, gray
        x[index] = im2
    return x


def normalizer(x):
    import numpy as np
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    x = (x - x_min) / (x_max - x_min)
    return x


def pre_process(features, labels, is_train=False):
    from sklearn.utils import shuffle
    assert (len(features) == len(labels))
    features = grayscale(features)
    features = normalizer(features)
    if is_train:
        features, labels = shuffle(features, labels)
    return features, labels
```

| **Property**          | **Value** |
|-----------------------|-----------|
| Before Normalization: |           |
| Pixel Value           | 0 to 255  |
| After Normalization:  |           |
| Pixel Value           | \-1 to +1 |
| Mean                  | \~0       |

 

![png](documentation/output_25_0.png)

![](documentation/output_27_0.png)

 

### **Image Transformations and Rotations:**

 

The idea for Label preserving data augmentation came from the [AlexNet for
ImageNet Classification](https://goo.gl/i8MHfX)

As we saw earlier, the dataset doesn’t contain the uniform distribution of the
samples for each class. We can fix it by generating new images by performing
transformation using translation, rotation, changing brightness etc. This is
called Data Augmentation.

With augmentation, we gain another advantage that now our training set is larger
than before and also more varied so it also helps in reducing the overfit during
the training process.

I primarily used OpenCV for image transformations.

```python
def visualize_augmented_features(features, labels, index, images_in_row=1):
    import matplotlib.pyplot as plt
    from random import choice
    %matplotlib inline
    indices = get_classes_samples(index, labels)
    fig, axes = plt.subplots(1, images_in_row, figsize=(15, 15))
    for index in range(images_in_row):
        random_index = choice(indices)
        image = features[random_index].squeeze()
        axes[index].imshow(image)
    plt.show()
    
    
def perform_rotation(image, cols, rows):
    from random import randint
    import cv2
    center = (int(cols / 2), int(cols / 2))
    angle = randint(-12, 12)
    transformer = cv2.getRotationMatrix2D(center, angle, 1)
    image = cv2.warpAffine(image, transformer, (cols, rows))
    return image


def perform_translation(image, cols, rows, value):
    import cv2
    import numpy as np
    transformer = np.float32([[1, 0, value], [0, 1, value]])
    image = cv2.warpAffine(image, transformer, (cols, rows))
    return image

    
def perform_transformation(feature, label):
    from random import randint
    transform_level = 10
    rows, cols, channels = feature.shape
    rotational_value = randint(-int(rows / transform_level), int(rows / transform_level))
    image = perform_rotation(feature, cols, rows)
    image = perform_translation(image, cols, rows, rotational_value)
    return image, label


def augment_dataset(features, labels, n_classes):
    from random import randint
    from sklearn.utils import shuffle
    import numpy as np
    transforms_per_image = 20
    iterations = 100
    augmented_features, augmented_labels = [], []
    for _i_ in range(iterations):
        for i in range(transforms_per_image):
            # get a random class from 0 to 42
            random_class = randint(0, n_classes)
            # select 10 features and labels of that class
            selected_index = get_classes_samples(random_class, labels [random_class:random_class + 1]
            # print("index: ", selected_index)
            selected_labels = labels[selected_index]
            # perform transformation in each of the features
            for index, transform_y in zip(selected_index, selected_labels):
                # get rows and cols of the image
                transform_x = features[index]
                rows, cols, channels = transform_x.shape
                # create several transforms from a single image
                for value in range(-int(rows), int(rows), 4):
                    # perform transformations on the image
                    aug_x, aug_y = perform_transformation(transform_x, transform_y)
                    augmented_features.append(aug_x)
                    augmented_labels.append(aug_y)
    # append the results of transformations
    augmented_features, augmented_labels = shuffle(augmented_features, augmented_labels)
    augmented_features = np.array(augmented_features)
    # assertion
    assert (len(augmented_features) == len(augmented_labels))
    return augmented_features, augmented_labels
```

#### Visualize how the transformation is performed.

 

![png](documentation/output_31_0.png)

![png](documentation/output_31_1.png)

![](documentation/output_31_2.png)

 

### **Design and Test a Model Architecture**

 

Now, I implemented the multi-layer Convolutional Neural Network architecture. 

 

The starting point was the LeNet Architecture which consists of Convolution
Layers followed by Fully Connected layers:

 

![](documentation/lenet.png)

Then I adjusted the architecture by following Sermanet & LeCun Publication on
Traffic Sign Recognition.   
[<http://yann.lecun.org/exdb/publis/psgz/sermanet-ijcnn-11.ps.gz>] 

 

![](documentation/multiscale-cnn.png)

 

Then I made tweak the architecture a little further by:

-   Using 3 Convolution layers instead of 2 layers.

-   Using Dropouts after the Fully Connected layers. Dropout was proposed by
    [Geoffrey Hinton et al](https://goo.gl/Y7QH0b). It is a technique to reduce
    overfit by randomly dropping the few units so that the network can never
    rely on any given activation. Dropout helps network to learn redundant
    representation of everything to make sure some of the information retain.

 

*Note: When Dropout technique is used, the dropped out neurons do not contribute
in forward and backward pass.*

```python
def le_net(_x_, mu, stddev, dropouts, input_channels=1, output_channels=10):
    from tensorflow.contrib.layers import flatten
    train_dropouts = {
        'c1': dropouts[0],
        'c2': dropouts[1],
        'c3': dropouts[2],
        'fc1': dropouts[3],
        'fc2': dropouts[4],
    }
    w, b = get_weights_biases(mu, stddev, input_channels, output_channels)
    padding = 'VALID'
    k = 2
    st, pool_st, pool_k = [1, 1, 1, 1], [1, k, k, 1], [1, k, k, 1]
    # Layer 1 -- convolution layer:
    conv1 = convolution_layer(_x_, w['c1'], b['c1'], st, padding, pool_k, pool_st, train_dropouts['c1'])
    # Layer 2 -- convolution layer:
    conv2 = convolution_layer(conv1, w['c2'], b['c2'], st, padding, pool_k, pool_st, train_dropouts['c2'])
    # Layer 3 -- convolution layer
    conv3 = convolution_layer(conv2, w['c3'], b['c3'], st, padding, pool_k, pool_st, train_dropouts['c3'], apply_pooling=False)
    # Flatten
    fc1 = flatten(conv3)
    print("Faltten: {}\n".format(fc1.get_shape()[1:]))
    # Layer 4 -- fully connected layer:
    fc1 = full_connected_layer(fc1, w['fc1'], b['fc1'], train_dropouts['fc1'])
    # Layer 5 -- full connected layer:
    fc2 = full_connected_layer(fc1, w['fc2'], b['fc2'], train_dropouts['fc2'])
    # Layer 6 -- fully connected output layer:
    out = output_layer(fc2, w['out'], b['out'])
    # parameters in each layer
    n_parameters(conv1, conv2, conv3, fc1, fc2, out)
    return out
```

My architecture is slightly modified from the above mentioned reference
architecture and is as follows: 

| **Layer**   | **Description** | **Filter Weight** | **Filter Bias** | **Stride** | **Padding** | **Dropout** | **Dimension**        | **Parameter** |
|-------------|-----------------|-------------------|-----------------|------------|-------------|-------------|----------------------|---------------|
| **Layer 1** | Convolutional   | 5x5x6             | 6               | 2x2        | Valid       | 1.0         | Input: 32x32x3       | 456           |
|             |                 |                   |                 |            |             |             | ReLU: 28x28x6        |               |
|             |                 |                   |                 |            |             |             | Max Pooling: 14x14x6 |               |
| **Layer 2** | Convolutional   | 5x5x16            | 16              | 2x2        | Valid       | 1.0         | Input: 14x14x6       | 2416          |
|             |                 |                   |                 |            |             |             | ReLU: 10x10x16       |               |
|             |                 |                   |                 |            |             |             | Max Pooling: 5x5x16  |               |
| **Layer 3** | Convolutional   | 5x5x400           | 400             | 2x2        | Valid       | 1.0         | Input: 5x5x16        | 160400        |
|             |                 |                   |                 |            |             |             | ReLU: 1x1x400        |               |
| **Flatten** |                 |                   |                 |            |             |             | 400                  |               |
| **Layer 4** | Fully Connected | 400x120           | 120             |            |             | 0.6         | Input: 400           | 48120         |
|             |                 |                   |                 |            |             |             | ReLU: 120            |               |
| **Layer 5** | Fully Connected | 120x84            | 84              |            |             | 0.5         | Input: 120           | 14520         |
|             |                 |                   |                 |            |             |             | ReLU: 84             |               |
| **Layer 6** | Output          | 84x43             | 43              |            |             |             | Input: 84            | 7140          |
|             |                 |                   |                 |            |             |             | Logits: 84           |               |

 
```python
retrain_model = False
no_improvement_count = 0
prev_accuracy = 0
current_accuracy = 0
if retrain_model:
    with tf.Session() as sess:
        sess.run(init)
        print("Training....")
        for e in range(hyper_params['epoch']):
            # training the network
            prev_accuracy = current_accuracy
            x_train_p, y_train_p = shuffle(x_train_p, y_train_p)
            batches = get_batches(hyper_params['batch_size'], x_train_p, y_train_p)
            for batch_x, batch_y in batches:
                batch_x, batch_y = shuffle(batch_x, batch_y)
                sess.run(optimizer, feed_dict={
                    x: batch_x, y: batch_y,
                    dropouts: hyper_params['dropouts']
                })
            # validation the network
            validation_accuracy = sess.run(accuracy, feed_dict={
                x: x_validation_p, y: y_validation_p,
                dropouts: hyper_params['test_dropouts']
            })
            # early termination
            current_accuracy = validation_accuracy
            no_improvement_count = no_improvement_count + 1 if current_accuracy < prev_accuracy else 0
            if no_improvement_count > 3:
                break
            print("{}th epoch - before: {:2.3f}%".format(e + 1, validation_accuracy * 100))
        saver.save(sess, save_file)
    print("Model saved")
```
The hyper parameters are as follows:

| **Parameter**      | **Value**    |
|--------------------|--------------|
| Mean               | 0            |
| Standard Deviation | 0.1          |
| Epochs             | 25           |
| Batch Size         | 128          |
| Learn Rate         | 0.001        |
| Dropouts           | Layer 1: 1.0 |
|                    | Layer 2: 1.0 |
|                    | Layer 3: 0.6 |
|                    | Layer 4: 0.5 |
|                    | Layer 5: 0.5 |
| Test Dropouts      | 1.0          |

 
When training a network on not-so-powerful computers, it is important to apply Mini-batching so that the network can be trained with small chunks of the training data at a time without overloading the memory of the machine. I wrote following code for mini-batching:

```python
def get_batches(_batch_size_, _features_, _labels_):
    import math
    total_size, index, batch = len(_features_), 0, []
    n_batches = int(math.ceil(total_size / _batch_size_)) if _batch_size_ > 0 else 0
    for _i_ in range(n_batches - 1):
        batch.append([_features_[index:index + _batch_size_],
                      _labels_[index:index + _batch_size_]])
        index += _batch_size_
    batch.append([_features_[index:], _labels_[index:]])
    return batch
```

Now we will train the classifier.

 

I used Adam Optimizer to optimize Weights and Biases using Back Propogation
instead of using Stochastic Gradient Descent. Following is the implementation of
the LeNet Architecture. For implementation details of each layer, please have a
look to the [Github
repo.](https://github.com/aurangzaib/CarND-Traffic-Sign-Classifier-Project)

 

When training a network on not-so-powerful computers, it is important to apply
Mini-batching so that the network can be trained with small chunks of the
training data at a time without overloading the memory of the machine. I wrote
following code for mini-batching:

 

Now, for actual training of the network, we need to create a session of
TensorFlow and optimize the parameters. My results for the Validation sets are:

 

| **Epochs** | **Accuracy (%)** | **Epochs** | **Accuracy (%)** |
|------------|------------------|------------|------------------|
| 1st        | 71.020           | 2nd        | 80.839           |
| 3rd        | 86.576           | 4th        | 86.469           |
| 5th        | 89.138           | 6th        | 90.680           |
| 7th        | 91.270           | 8th        | 92.449           |
| 9th        | 92.426           | 10th       | 93.424           |
| 11th       | 93.832           | 12th       | 93.379           |
| 13th       | 93.469           | 14th       | 94.490           |
| 15th       | 93.628           | 16th       | 94.376           |
| 17th       | 93.129           | 18th       | 94.535           |
| 19th       | 94.376           | 20         | 95.215           |
| 21st       | 94.921           | 22nd       | 94.671           |
| 23rd       | 94.649           | 24th       | 94.581           |
| 25th       | 94.172           |            |                  |

 

Now comes the part where we test the accuracy of the network on the hidden test
data.

```python
test_data = True
if test_data:
    with tf.Session() as sess:
        saver.restore(sess, save_file)
        print("Model restored")
        test_accuracy = sess.run(accuracy, feed_dict={
            x: x_test_p,
            y: y_test_p,
            dropouts: hyper_params['test_dropouts']
        })
        print("test accuracy: {:2.3f}%".format(test_accuracy * 100))
```

**The network is able to achieve 95.306% accuracy on the test data.**

| Test Accuracy | 95.306% |
|---------------|---------|


 

### **Test a Model on New Images:**

To give myself more insight into how your model is working, I downloaded several
images from the internet of traffic signs and tested the accuracy of the
pre-trained network.

![](documentation/output_58_0.png)

![](documentation/output_59_0.png)

 

**Discussion​ on New Test Data:**

 

**Image 1:**

It is Left Turn sign and the network classifies correctly. \<br /\>

 

**Image 2:**

It is a Bicycle Crossing sign but it is slightly modified from the sign the
network was trained on. The network confuses it with the Right-of-way sign.

 

**Image 3:**

It is Ahead Only sign and the network classifies correctly.

 

**Image 4:**

It is a Traffic Signal sign. The network confuses it with the Pedastrian sign.

 

**Image 5:**

It is a Slippery Road sign. The network confuses it with the Stop sign. The
reason might be that in train data the Slippery Road sign has car inclined in it
while the test image has car horizontal it.

 

**Image 6, 7, 8:**

These are Priority road, Turn Right Ahead and Yeild signs respectively. The
network classifies correclty.

```python
test_new_data = True
if test_new_data:
    with tf.Session() as sess:
        x_test_new, y_test_new, file_names = get_new_test_data('/test-data/')
        x_test_new_p, y_test_new_p = pre_process(x_test_new, y_test_new)
        saver.restore(sess, save_file)
        predicted_logits = sess.run(accuracy, feed_dict={
            x: x_test_new_p,
            y: y_test_new_p,
            dropouts: hyper_params['test_dropouts']
        })
        prediction_probabilities = sess.run(soft_max_prb, feed_dict={
            x: x_test_new_p,
            dropouts: hyper_params['test_dropouts']
        })
        print("predicted logits: {}\n".format(predicted_logits))
        top_p, top_i = sess.run(tf.nn.top_k(tf.constant(prediction_probabilities), k=5))
        for index in range(len(top_p)):
            visualize_test_images(x_test_new[index], is_single=True)
            print(file_names[index])
            for p, i in zip(top_p[index], top_i[index]):
                print("{}:  {:2.3f}%".format(traffic_sign_name(i), p * 100))
            print("\n")
```

**Predictions on New Test Data:**

![](documentation/output_65_3.png)

| **Predictions**                       | **Confidence (%)**       |
|---------------------------------------|--------------------------|
| Dangerous curve to the left           | 100.000                  |
| Slippery road                         | 0.000                    |
| Right-of-way at the next intersection | 0.000                    |
| Double curve                          | 0.000                    |
| Road work                             | 0.000                    |
| **Ground Truth**                      | **Dangerous curve left** |

![](documentation/output_65_5.png)

| **Predictions**                       | **Confidence (%)**   |
|---------------------------------------|----------------------|
| Road narrows on the right             | 42.401               |
| Pedestrians                           | 37.532               |
| General caution                       | 19.522               |
| Traffic signals                       | 0.370                |
| Right-of-way at the next intersection | 0.150                |
| **Ground Truth**                      | **Bicycle crossing** |

![](documentation/output_65_9.png)

| **Predictions**      | **Confidence (%)** |
|----------------------|--------------------|
| Ahead only           | 100.000            |
| No passing           | 0.000              |
| Bicycles crossing    | 0.000              |
| Turn left ahead      | 0.000              |
| Speed limit (60km/h) | 0.000              |
| **Ground Truth**     | **Ahead Only**     |

![](documentation/output_65_11.png)

| **Predictions**           | **Confidence (%)**  |
|---------------------------|---------------------|
| Road narrows on the right | 98.333%             |
| Pedestrians               | 0.889               |
| Road work                 | 0.563               |
| Children crossing         | 0.071               |
| General caution           | 0.052               |
| **Ground Truth**          | **Traffic Signals** |

![](documentation/output_65_21.png)

| **Prediction**       | **Confidence (%)** |
|----------------------|--------------------|
| No entry             | 99.996             |
| Stop                 | 0.004              |
| Turn right ahead     | 0.000              |
| Roundabout mandatory | 0.000              |
| Speed limit (70km/h) | 0.000              |
| **Ground Truth**     | **Slippery road**  |

![](documentation/output_65_27.png)

| **Predictions**                       | **Confidence (%)** |
|---------------------------------------|--------------------|
| Priority road                         | 100.000            |
| Roundabout mandatory                  | 0.000              |
| Right-of-way at the next intersection | 0.000              |
| Yield                                 | 0.000              |
| No passing                            | 0.000              |
| **Ground Truth**                      | **Priority road**  |

![](documentation/output_65_29.png)

| **Predictions**      | **Confidence (%)**       |
|----------------------|--------------------------|
| Turn right ahead     | 99.983                   |
| Stop                 | 0.017                    |
| Keep left            | 0.000                    |
| Speed limit (30km/h) | 0.000                    |
| No entry             | 0.000                    |
| **Ground Truth**     | **Go straight or right** |

![](documentation/output_65_31.png)

| **Predictions**                                    | **Confidence (%)**       |
|----------------------------------------------------|--------------------------|
| Roundabout mandatory                               | 42.161                   |
| End of no passing by vehicles over 3.5 metric tons | 38.446                   |
| Right-of-way at the next intersection              | 13.663                   |
| Priority road                                      | 2.242                    |
| Keep right                                         | 1.334                    |
| **Ground Truth**                                   | **Roundabout mandatory** |

 
=

*For the complete implementation and notebook:*

<https://github.com/aurangzaib/CarND-Traffic-Sign-Classifier-Project>
