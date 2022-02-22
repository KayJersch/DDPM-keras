from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, UpSampling2D, Concatenate

def build_unet(num_step):
    unet_input = Input(shape=(28,28,1))
    step_input = Input(shape=(num_step))

    unet_conv1 = Conv2D(32,3,padding='same',activation='elu')(unet_input)
    unet_pool1 = MaxPooling2D(padding='same')(unet_conv1)

    unet_conv2 = Conv2D(64,3,padding='same',activation='elu')(unet_pool1)
    unet_pool2 = MaxPooling2D(padding='same')(unet_conv2)

    unet_conv3 = Conv2D(128,3,padding='same',activation='elu')(unet_pool2)

    unet_flat = Flatten()(unet_conv3)
    unet_dense1 = Dense(7*7*128,'relu')(unet_flat)

    step_dense = Dense(7*7*128,'relu')(step_input)

    unet_concat1 = Concatenate()([unet_dense1,step_dense])
    unet_dense2 = Dense(7*7*128,'relu')(unet_concat1)

    unet_resh = Reshape((7,7,128))(unet_dense2)

    unet_conv4 = Conv2D(128,3,padding='same',activation='elu')(unet_resh)
    unet_concat2 = Concatenate(axis=-1)([unet_conv4, unet_conv3])
    unet_upsamp1 = UpSampling2D()(unet_concat2)

    unet_conv5 = Conv2D(64,3,padding='same',activation='elu')(unet_upsamp1)
    unet_concat3 = Concatenate(axis=-1)([unet_conv5, unet_conv2])
    unet_upsamp2 = UpSampling2D()(unet_concat3)

    unet_conv6 = Conv2D(32,3,padding='same',activation='elu')(unet_upsamp2)
    unet_concat4 = Concatenate(axis=-1)([unet_conv6, unet_conv1])

    unet_conv5 = Conv2D(1,3,padding='same',activation='elu')(unet_concat4)

    unet = Model(inputs=[unet_input,step_input],outputs=unet_conv5)
    unet.compile('sgd','mse')

    return unet