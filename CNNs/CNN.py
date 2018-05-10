import tensorflow as tf
import tflearn

def getCNN(classes, finetune=False):
    """
        This is the current working CNN.
        classes is the number of classes (neurons in the final softmax layer) to be processed.
        If finetune==True, only allow the final two levels to be trainable.
    """
    # Neural net (two-channel)
    tf.reset_default_graph()
    tflearn.initializations.normal()

    if finetune == True:
        # Input layer:
        inp = tflearn.layers.core.input_data(shape=[None,34,34,34,2])

        # First layer:
        conv_0 = tflearn.layers.conv.conv_3d(inp, 32, [10,10,10],  activation="leaky_relu", trainable=False)
        max_0 = tflearn.layers.conv.max_pool_3d(conv_0, [2,2,2], strides=[2,2,2])

        # Second layer:
        conv_1 = tflearn.layers.conv.conv_3d(max_0, 64, [5,5,5],  activation="leaky_relu", trainable=False)
        max_1 = tflearn.layers.conv.max_pool_3d(conv_1, [2,2,2], strides=[2,2,2])

        # Third layer:
        conv_2 = tflearn.layers.conv.conv_3d(max_1, 128, [2,2,2], activation="leaky_relu", trainable=False) # This was added for CNN 2017-07-28
        max_2 = tflearn.layers.conv.max_pool_3d(conv_2, [2,2,2], strides=[2,2,2]) # This was added for CNN 2017-08-24

        #Fourth layer:
        conv_3 = tflearn.layers.conv.conv_3d(max_2, 256, [2,2,2], activation="leaky_relu", trainable=False) # This was added for CNN 2017-08-24
        #net = tflearn.layers.core.fully_connected(net, 4096, activation="leaky_relu", trainable=False)#, regularizer="L2", weight_decay=0.0001) # This was added for CNN 2017-08-24

        # Fully connected layers
        fc_0 = tflearn.layers.core.fully_connected(conv_3, 2048, activation="leaky_relu", regularizer="L2", weight_decay=0.0001, trainable=False)
        #net = tflearn.layers.core.dropout(net, keep_prob=0.5)

        fc_1 = tflearn.layers.core.fully_connected(fc_0, 1024, activation="leaky_relu", regularizer="L2", weight_decay=0.0001, trainable=False)
        #net = tflearn.layers.core.dropout(net, keep_prob=0.5)

        fc_2 = tflearn.layers.core.fully_connected(fc_1, 512, activation="leaky_relu", regularizer="L2", weight_decay=0.0001)
        #net = tflearn.layers.core.dropout(net, keep_prob=0.5)

        # Output layer:
        fc_3 = tflearn.layers.core.fully_connected(fc_2, classes, activation="softmax")

        outp = tflearn.layers.estimator.regression(fc_3, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy')
        model = tflearn.DNN(outp, tensorboard_verbose=0)

        return model

    else:
        # Input layer:
        inp = tflearn.layers.core.input_data(shape=[None,34,34,34,2])

        # First layer:
        conv_0 = tflearn.layers.conv.conv_3d(inp, 32, [10,10,10],  activation="leaky_relu")
        max_0 = tflearn.layers.conv.max_pool_3d(conv_0, [2,2,2], strides=[2,2,2])

        # Second layer:
        conv_1 = tflearn.layers.conv.conv_3d(max_0, 64, [5,5,5],  activation="leaky_relu")
        max_1 = tflearn.layers.conv.max_pool_3d(conv_1, [2,2,2], strides=[2,2,2])

        # Third layer:
        conv_2 = tflearn.layers.conv.conv_3d(max_1, 128, [2,2,2], activation="leaky_relu") # This was added for CNN 2017-07-28
        max_2 = tflearn.layers.conv.max_pool_3d(conv_2, [2,2,2], strides=[2,2,2]) # This was added for CNN 2017-08-24

        #Fourth layer:
        conv_3 = tflearn.layers.conv.conv_3d(max_2, 256, [2,2,2], activation="leaky_relu") # This was added for CNN 2017-08-24
        #net = tflearn.layers.core.fully_connected(net, 4096, activation="leaky_relu")#, regularizer="L2", weight_decay=0.0001) # This was added for CNN 2017-08-24

        # Fully connected layers
        fc_0 = tflearn.layers.core.fully_connected(conv_3, 2048, activation="leaky_relu", regularizer="L2", weight_decay=0.0001)
        #net = tflearn.layers.core.dropout(net, keep_prob=0.5)

        fc_1 = tflearn.layers.core.fully_connected(fc_0, 1024, activation="leaky_relu", regularizer="L2", weight_decay=0.0001)
        #net = tflearn.layers.core.dropout(net, keep_prob=0.5)

        fc_2 = tflearn.layers.core.fully_connected(fc_1, 512, activation="leaky_relu", regularizer="L2", weight_decay=0.0001)
        #net = tflearn.layers.core.dropout(net, keep_prob=0.5)

        # Output layer:
        fc_3 = tflearn.layers.core.fully_connected(fc_2, classes, activation="softmax")

        outp = tflearn.layers.estimator.regression(fc_3, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy')
        model = tflearn.DNN(outp, tensorboard_verbose=0)

        return model
