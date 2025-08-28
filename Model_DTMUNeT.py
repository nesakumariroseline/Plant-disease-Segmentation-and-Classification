from tensorflow import keras
from keras.layers import Conv2D, Input, MaxPooling2D, Conv2DTranspose, concatenate, Reshape, Permute, Dense, Dropout, \
    LayerNormalization
from keras.models import Model
import numpy as np

# Define the Transformer encoder block
class TransformerEncoderBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([Dense(ff_dim, activation='relu'), Dense(embed_dim)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=True):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# Define the Trans MobileUNet++ model
def TransMobileUNetPlusPlus(input_shape, num_classes):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Transformer Encoder
    embed_dim = 128
    num_heads = 8
    ff_dim = 256
    transformer_input = Reshape((-1, embed_dim))(pool3)
    transformer_output = transformer_input
    for _ in range(4):  # Number of transformer blocks
        transformer_output = TransformerEncoderBlock(embed_dim, num_heads, ff_dim)(transformer_output)

    # Decoder
    transformer_output = Reshape(pool3.shape[1:])(transformer_output)
    upconv3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(transformer_output)
    upconv3 = concatenate([upconv3, conv3], axis=3)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(upconv3)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)

    upconv2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv4)
    upconv2 = concatenate([upconv2, conv2], axis=3)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(upconv2)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)

    upconv1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv5)
    upconv1 = concatenate([upconv1, conv1], axis=3)
    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(upconv1)
    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv6)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(conv6)

    model = Model(inputs, outputs)

    return model


def Model_TransMobileUNetplusplus(Data, Target):
    input_shape = (256, 256, 3)  # Adjust input shape as needed
    num_classes = 3  # Modify the number of classes

    Train_Temp = np.zeros((Data.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(Data.shape[0]):
        Train_Temp[i, :] = np.resize(Data[i], (input_shape[0], input_shape[1], input_shape[2]))
    Train_X = Train_Temp.reshape(Train_Temp.shape[0], input_shape[0], input_shape[1], input_shape[2])

    Test_Temp = np.zeros((Target.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(Target.shape[0]):
        Test_Temp[i, :] = np.resize(Target[i], (input_shape[0], input_shape[1], input_shape[2]))
    Train_Y = Test_Temp.reshape(Test_Temp.shape[0], input_shape[0], input_shape[1], input_shape[2])

    model = TransMobileUNetPlusPlus(input_shape, num_classes)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Train_X, Train_Y, epochs=150)
    Image = model.predict(Data)
    return Image


