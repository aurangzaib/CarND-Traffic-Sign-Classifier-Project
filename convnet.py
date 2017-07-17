hyper_params = {
    "mu": 0,
    "stddev": 0.1,
    "epoch": 25,
    "batch_size": 128,
    "rate": 0.001,
    "dropouts": [1., 1., .6, .5, .5],
    "test_dropouts": [1., 1., 1., 1., 1.]
}


def get_weights_biases(mu, sd, input_channels, output_channels):
    print("Output channel: {}\n".format(output_channels))
    import tensorflow as tf
    """
    tensorflow filter size formula for valid padding:
            Hf = H - Ho*Hs + 1
            Wf = W - Wo*Ws + 1
            Df = K
    """
    w = {
        'c1': tf.Variable(tf.truncated_normal([5, 5, input_channels, 6], mean=mu, stddev=sd)),
        'c2': tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean=mu, stddev=sd)),
        'c3': tf.Variable(tf.truncated_normal([5, 5, 16, 400], mean=mu, stddev=sd)),
        'fc1': tf.Variable(tf.truncated_normal([400, 120], mean=mu, stddev=sd)),
        'fc2': tf.Variable(tf.truncated_normal([120, 84], mean=mu, stddev=sd)),
        'out': tf.Variable(tf.truncated_normal([84, output_channels], mean=mu, stddev=sd)),
    }
    b = {
        'c1': tf.Variable(tf.truncated_normal([6], mean=mu, stddev=sd)),
        'c2': tf.Variable(tf.truncated_normal([16], mean=mu, stddev=sd)),
        'c3': tf.Variable(tf.truncated_normal([400], mean=mu, stddev=sd)),
        'fc1': tf.Variable(tf.truncated_normal([120], mean=mu, stddev=sd)),
        'fc2': tf.Variable(tf.truncated_normal([84], mean=mu, stddev=sd)),
        'out': tf.Variable(tf.truncated_normal([output_channels], mean=mu, stddev=sd))
    }
    return w, b


def convolution_layer(x, w, b, st, padding, pool_k, pool_st, dropout, apply_pooling=True):
    import tensorflow as tf
    print("Conv - input: {}".format(x.get_shape()))
    conv = tf.nn.conv2d(x, filter=w, strides=st, padding=padding)
    conv = tf.nn.bias_add(conv, bias=b)
    conv = tf.nn.relu(conv)
    print("Conv - relu: {}".format(conv.get_shape()))
    if apply_pooling:
        """
        max pooling reduces total # of parameters and reduces overfitting
        dropout is preferred over max pooling
        max pooling causes information loss
        """
        conv = tf.nn.max_pool(conv, ksize=pool_k, strides=pool_st, padding=padding)
    conv = tf.nn.dropout(conv, keep_prob=dropout)
    print("Conv - dropout: {}\n".format(conv.get_shape()))
    return conv


def full_connected_layer(fc, w, b, dropout):
    import tensorflow as tf
    print("FC - input: {}".format(fc.get_shape()))
    fc = tf.add(tf.matmul(fc, w), b)
    fc = tf.nn.relu(fc)
    print("FC - relu: {}".format(fc.get_shape()))
    fc = tf.nn.dropout(fc, keep_prob=dropout)
    print("FC - dropout: {}\n".format(fc.get_shape()))
    return fc


def output_layer(fc, w, b):
    import tensorflow as tf
    print("Out - input: {}".format(fc.get_shape()))
    out = tf.add(tf.matmul(fc, w), b)
    print("Out - logits: {}".format(fc.get_shape()))
    return out


def n_parameters(layer1, layer2, layer3, layer4, layer5, layer6):
    """
     without parameter sharing:
        num_params = (output)*(filter) + (output)*(bias)
        for example  --> num_params = (14x14x20)*(8x8x3)  + (14z14x20)*(1)

    with parameter sharing:
        num_params = (output_depth)*(filter) + (output_depth)*(bias)
        for example --> num_params = (20)*(8x8x3)  + (20)*(1)
    """
    # parameter sharing is assumed
    dim = layer1.get_shape()[3]
    layer1_params = dim * (5 * 5 * 3) + dim * 1
    dim = layer2.get_shape()[3]
    layer2_params = dim * (5 * 5 * 6) + dim * 1
    dim = layer3.get_shape()[3]
    layer3_params = dim * (5 * 5 * 16) + dim * 1
    dim = layer4.get_shape()[1]
    layer4_params = (dim * 400) + dim * 1
    dim = layer4.get_shape()[1]
    layer5_params = (dim * 120) + dim * 1
    dim = layer5.get_shape()[1]
    layer6_params = (dim * 84) + dim * 1
    total_params = layer1_params + layer2_params + layer3_params + layer4_params + layer5_params + layer6_params

    print("Layer 1 Params: {}".format(layer1_params))
    print("Layer 2 Params: {}".format(layer2_params))
    print("Layer 3 Params: {}".format(layer3_params))
    print("Layer 4 Params: {}".format(layer4_params))
    print("Layer 5 Params: {}".format(layer5_params))
    print("Layer 6 Params: {}".format(layer6_params))
    print("Total Params:   {}".format(total_params))


def le_net(_x_, mu, stddev, dropouts, input_channels=1, output_channels=10):
    from tensorflow.contrib.layers import flatten
    train_dropouts = {'c1': dropouts[0], 'c2': dropouts[1], 'c3': dropouts[2], 'fc1': dropouts[3], 'fc2': dropouts[4]}
    w, b = get_weights_biases(mu, stddev, input_channels, output_channels)
    padding = 'VALID'
    k = 2
    st, pool_st, pool_k = [1, 1, 1, 1], [1, k, k, 1], [1, k, k, 1]
    # Layer 1 -- convolution layer:
    conv1 = convolution_layer(_x_, w['c1'], b['c1'], st, padding, pool_k, pool_st, train_dropouts['c1'])
    # Layer 2 -- convolution layer:
    conv2 = convolution_layer(conv1, w['c2'], b['c2'], st, padding, pool_k, pool_st, train_dropouts['c2'])
    # Layer 3 -- convolution layer
    conv3 = convolution_layer(conv2, w['c3'], b['c3'], st, padding, pool_k, pool_st, train_dropouts['c3'],
                              apply_pooling=False)
    # Flatten
    fc1 = flatten(conv3)
    print("Flatten: {}\n".format(fc1.get_shape()))
    # Layer 3 -- fully connected layer:
    fc1 = full_connected_layer(fc1, w['fc1'], b['fc1'], train_dropouts['fc1'])
    # Layer 4 -- full connected layer:
    fc2 = full_connected_layer(fc1, w['fc2'], b['fc2'], train_dropouts['fc2'])
    # Layer 5 -- fully connected output layer:
    out = output_layer(fc2, w['out'], b['out'])
    # parameters in each layer
    n_parameters(conv1, conv2, conv3, fc1, fc2, out)
    return out
