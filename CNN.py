import tensorflow as tf
import tflearn

def getCNN():
    """
        This is the current working CNN.
    """
    # Neural net (two-channel)
    tf.reset_default_graph()
    tflearn.initializations.normal()

    # Input layer:
    net = tflearn.layers.core.input_data(shape=[None,34,34,34,2])

    # First layer:
    net = tflearn.layers.conv.conv_3d(net, 32, [10,10,10],  activation="leaky_relu")
    net = tflearn.layers.conv.max_pool_3d(net, [2,2,2], strides=[2,2,2])

    # Second layer:
    net = tflearn.layers.conv.conv_3d(net, 64, [5,5,5],  activation="leaky_relu")
    net = tflearn.layers.conv.max_pool_3d(net, [2,2,2], strides=[2,2,2])

    # Third layer:
    net = tflearn.layers.conv.conv_3d(net, 128, [2,2,2], activation="leaky_relu") # This was added for CNN 2017-07-28
    net = tflearn.layers.conv.max_pool_3d(net, [2,2,2], strides=[2,2,2]) # This was added for CNN 2017-08-24

    #Fourth layer:
    net = tflearn.layers.conv.conv_3d(net, 256, [2,2,2], activation="leaky_relu") # This was added for CNN 2017-08-24
    net = tflearn.layers.core.fully_connected(net, 4096, activation="leaky_relu")#, regularizer="L2", weight_decay=0.0001) # This was added for CNN 2017-08-24

    # Fully connected layers
    net = tflearn.layers.core.fully_connected(net, 2048, activation="leaky_relu")#, regularizer="L2", weight_decay=0.0001)
    #net = tflearn.layers.core.dropout(net, keep_prob=0.5)

    net = tflearn.layers.core.fully_connected(net, 1024, activation="leaky_relu")#, regularizer="L2", weight_decay=0.0001)
    #net = tflearn.layers.core.dropout(net, keep_prob=0.5)

    net = tflearn.layers.core.fully_connected(net, 512, activation="leaky_relu")#, regularizer="L2", weight_decay=0.0001)
    #net = tflearn.layers.core.dropout(net, keep_prob=0.5)

    # Output layer:
    net = tflearn.layers.core.fully_connected(net, 5, activation="softmax")

    net = tflearn.layers.estimator.regression(net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy')
    model = tflearn.DNN(net, tensorboard_verbose=0)

    return model
