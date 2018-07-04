import tensorflow as tf
import tflearn

def getCNN(classes, observe=False):
    """
        This is the current working CNN.
        classes is the number of classes (neurons in the final softmax layer) to be processed.
        If finetune==True, only allow the final two levels to be trainable.
    """
    # Neural net (two-channel)
    tf.reset_default_graph()
    tflearn.initializations.normal()

    # leaky_relu replaced with relu. Max pooling replaced with strides in conv layers. 2018-05-18

    # Input layer:
    inp = tflearn.layers.core.input_data(shape=[None,32,32,32,2])

    # First layer:
    conv_0 = tflearn.layers.conv.conv_3d(inp, 32, [4,4,4], strides=2,  activation="relu") # [16,16,16]

    # Second layer:
    conv_1 = tflearn.layers.conv.conv_3d(conv_0, 64, [4,4,4], strides=2, activation="relu") # [8,8,8]

    # Third layer:
    conv_2 = tflearn.layers.conv.conv_3d(conv_1, 128, [2,2,2], activation="relu")

    # Fourth layer:
    conv_3 = tflearn.layers.conv.conv_3d(conv_2, 256, [2,2,2], activation="relu")

    # Global pooling layer:
    global_pool_0 = tf.reduce_mean(conv_3, [1,2,3])

    # Output layer:
    fc_0 = tflearn.layers.core.fully_connected(global_pool_0, classes, activation="softmax")

    outp = tflearn.layers.estimator.regression(fc_0, optimizer="adam", learning_rate=0.0001, loss='categorical_crossentropy')

    model = tflearn.DNN(outp, tensorboard_verbose=0)

    if observe:
        return model, conv_3
    if not observe:
        return model
