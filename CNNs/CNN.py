import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv3D, GlobalAveragePooling3D, Dense

def getCNN(n_classes, observe=False):
    """
        This is the current working CNN.
        classes is the number of classes (neurons in the final softmax layer) to be processed.
        If finetune==True, only allow the final two levels to be trainable.
    """
    # Neural net (two-channel)
    # leaky_relu replaced with relu. Max pooling replaced with strides in conv layers. 2018-05-18
    inp = Input(shape=(32,32,32,2))

    # First layer:
    conv_0 = Conv3D(32, [4,4,4], strides=2,  activation="relu")(inp) # [16,16,16]

    # Second layer:
    conv_1 = Conv3D(64, [4,4,4], strides=2, activation="relu")(conv_0) # [8,8,8]

    # Third layer:
    conv_2 = Conv3D(128, [2,2,2], activation="relu")(conv_1)

    # Fourth layer:
    conv_3 = Conv3D(256, [2,2,2], activation="relu")(conv_2)

    # Global pooling layer:
    global_pool_0 = GlobalAveragePooling3D()(conv_3)

    # Output layer:
    fc_0 = Dense(n_classes, activation='softmax')(global_pool_0)

    model = Model(inputs=inp, outputs=fc_0)
    return model
