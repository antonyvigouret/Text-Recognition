import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Bidirectional,
    LSTM,
    MaxPooling2D,
    BatchNormalization,
    Activation,
    Input,
    Lambda,
    Add,
)
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from utils import ALPHABET


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

### Subclass implementation of the CRNN model
class CRNN(tf.keras.Model):

    def __init__(self, alphabet, input_shape=(32, None, 3)):
        super(CRNN, self).__init__(name = "CRNN")
        self.alphabet = alphabet

        self.input_layer = Input(input_shape)

        self.conv1 = Conv2D(64, 3, padding="same", activation="relu", name="conv2d_1")
        self.max_pool1 = MaxPooling2D((2, 2), (2, 2), name="pool2d_1")

        self.conv2 = Conv2D(128, 3, padding="same", activation="relu", name="conv2d_2")
        self.max_pool2 = MaxPooling2D((2, 2), (2, 2), name="pool2d_2")

        self.conv3 = Conv2D(256, 3, padding="same", activation="relu", name="conv2d_3")

        self.conv4 = Conv2D(256, 3, padding="same", activation="relu", name="conv2d_4")
        self.max_pool4 = MaxPooling2D((2, 1), (2, 1), name="pool2d_4")

        self.conv5 = Conv2D(512, 3, padding="same", activation="relu", name="conv2d_5")
        self.batch_norm5 = BatchNormalization(name="batch_norm_5")

        self.conv6 = Conv2D(512, 3, padding="same", activation="relu", name="conv2d_6")
        self.batch_norm6 = BatchNormalization(name="batch_norm_6")

        self.max_pool6 = MaxPooling2D((2, 1), (2, 1), name="pool2d_6")

        self.conv7 = Conv2D(512, 2, padding="valid", activation="relu", name="conv2d_7")

        self.bidiLSTM1 = Bidirectional(LSTM(256, return_sequences=True), name="bidirectional_1")
        self.bidiLSTM2 = Bidirectional(LSTM(256, return_sequences=True), name="bidirectional_2")

        self.dense = Dense(len(self.alphabet) + 1)

        self.out = self.call(self.input_layer, training=False)

        super(CRNN, self).__init__(
            inputs=self.input_layer,
            outputs=self.out)
        

    def call(self, inputs, training=True):

        #[?, 32, W, 1] -> [?, 32, W, 64] -> [?, 16, W/2, 1] 
        x = self.conv1(inputs)
        x = self.max_pool1(x)

        #[?, 16, W/2, 1] -> [?, 16, W/2, 128] -> [?, 8, W/4, 128] 
        x = self.conv2(x)
        x = self.max_pool2(x)

        #[?, 8, W/4, 128] -> [?, 8, W/4, 256]
        x = self.conv3(x)

        #[?, 8, W/4, 256] -> [?, 8, W/2, 256] -> [?, 4, W/4, 256] 
        x = self.conv4(x)
        x = self.max_pool4(x)

        #[?, 4, W/4, 512] -> [?, 4, W/4, 512]
        x = self.conv5(x)
        x = self.batch_norm5(x)

        #[?, 4, W/4, 512] -> [?, 4, W/4, 512] -> [?, 2, W/4, 512]
        x = self.conv6(x)
        x = self.batch_norm6(x)
        x = self.max_pool6(x)

        # [?, 2, W/4, 512] -> [?, 1, W/4-3, 512] 
        x = self.conv7(x)

        x = tf.squeeze(x, axis=1)
        # [batch, width_seq, depth_chanel]

        x = self.bidiLSTM1(x)
        x = self.bidiLSTM2(x)

        logits = self.dense(x)

        y_pred = Activation("softmax", name="softmax")(logits)

        return y_pred

    
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x["the_input"], training=True) # Forward pass
            y_pred = y_pred[:, 2:, :]
            loss = tf.reduce_mean(ctc_lambda_func((y_pred, x["the_labels"], tf.reshape(x["input_length"], [-1, 1]), tf.reshape(x["label_length"], [-1, 1]))))
            print(loss)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))


    def build(self):
        # Initialize the graph
        self._is_graph_network = True
        self._init_graph_network(
            inputs=self.input_layer,
            outputs=self.out
        )




def get_CRNN(weights=None):
    input_data = Input(name="the_input", shape=(32, None, 3), dtype="float32")
    inner = Conv2D(32, 3, padding="same", kernel_initializer="he_normal", name="conv1")(
        input_data
    )
    inner = BatchNormalization()(inner)
    inner = Activation("relu")(inner)
    inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="max1")(inner)

    inner = Conv2D(64, 3, padding="same", kernel_initializer="he_normal", name="conv2")(
        inner
    )
    inner = BatchNormalization()(inner)
    inner = Activation("relu")(inner)
    inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="max2")(inner)

    inner = Conv2D(
        128, 3, padding="same", kernel_initializer="he_normal", name="conv3"
    )(inner)
    inner = BatchNormalization()(inner)
    inner = Activation("relu")(inner)
    inner = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name="max3")(inner)

    inner = Conv2D(
        256, 3, padding="same", kernel_initializer="he_normal", name="conv4"
    )(inner)
    inner = BatchNormalization()(inner)
    inner = Activation("relu")(inner)
    inner = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name="max4")(inner)

    inner = Conv2D(
        256, 3, padding="same", kernel_initializer="he_normal", name="conv5"
    )(inner)
    inner = BatchNormalization()(inner)
    inner = Activation("relu")(inner)
    inner = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name="max5")(inner)

    inner = tf.squeeze(inner, axis=1)

    # stack of 3 bidi lstm
    inner = Bidirectional(LSTM(256, return_sequences=True))(inner)
    inner = Bidirectional(LSTM(256, return_sequences=True))(inner)
    inner = Bidirectional(LSTM(512, return_sequences=True))(inner)

    # transforms RNN output to character activations:
    alphabet_size = 107
    inner = Dense(alphabet_size, kernel_initializer="he_normal", name="dense2")(inner)
    y_pred = Activation("softmax", name="softmax")(inner)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name="the_labels", shape=[None], dtype="float32")
    input_length = Input(name="input_length", shape=[1], dtype="int64")
    label_length = Input(name="label_length", shape=[1], dtype="int64")
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")(
        [y_pred, labels, input_length, label_length]
    )

    # clipnorm seems to speeds up convergence
    optimizer = Adam(learning_rate=0.001, decay=1e-6)

    model = Model(
        inputs=[input_data, labels, input_length, label_length], outputs=loss_out
    )

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={"ctc": lambda y_true, y_pred: y_pred}, optimizer=optimizer)
    if weights:
        model.load_weights(weights)
    # captures output of softmax so we can decode the output during visualization
    test_func = K.function([input_data], [y_pred])

    return model, test_func

### Keras functional API implementation of the CRNN model
def CRNN_model(weights=None):
    inputs = Input(name="the_input", shape=(32, None, 3), dtype="float32")

    x = Conv2D(64, 3, padding="same", name="conv2d_0")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D((2, 2), (2, 2), name="pool2d_0")(x)

    x = Conv2D(128, 3, padding="same", name="conv2d_1")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D((2, 2), (2, 2), name="pool2d_1")(x)

    x = Conv2D(256, 3, padding="same", name="conv2d_2")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D((2, 1), (2, 1), name="pool2d_2")(x)

    x = Conv2D(512, 3, padding="same", name="conv2d_3")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D((2, 1), (2, 1), name="pool2d_3")(x)

    x = Conv2D(512, 3, padding="same", name="conv2d_4")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D((2, 1), (2, 1), name="pool2d_4")(x)

    x = tf.squeeze(x, axis=1)
    # [batch width_seq depth_chanel]

    x = Bidirectional(LSTM(256, return_sequences=True), name="bidirectional_1")(x)
    x = Bidirectional(LSTM(256, return_sequences=True), name="bidirectional_2")(x)
    x = LSTM(512, return_sequences=True)(x)

    x = Dense(len(ALPHABET) + 1)(x)
    y_pred = Activation("softmax", name="softmax")(x)

    Model(inputs=inputs, outputs=y_pred).summary()

    labels = Input(name="the_labels", shape=[None], dtype="float32")
    input_length = Input(name="input_length", shape=[1], dtype="int64")
    label_length = Input(name="label_length", shape=[1], dtype="int64")

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")(
        [y_pred, labels, input_length, label_length]
    )

    sgd = Adam(learning_rate=0.001)

    model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={"ctc": lambda y_true, y_pred: y_pred}, optimizer=sgd)
    if weights:
        model.load_weights(weights)
    test_func = K.function([inputs], [y_pred])
    return model, test_func


def get_CResRNN(weights=None):
    inputs = Input(name="the_input", shape=(32, None, 3), dtype="float32")

    x = Conv2D(64, 7, padding="same", name="conv2d_0")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x_orig = x

    x = Conv2D(64, 3, padding="same", name="conv2d_0_1")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(64, 3, padding="same", name="conv2d_0_2")(inputs)
    x = BatchNormalization()(x)

    x = Add()([x, x_orig])
    x = Activation("relu")(x)

    x = MaxPooling2D((2, 2), (2, 2), name="pool2d_0")(x)

    

    x = Conv2D(128, 3, padding="same", name="conv2d_1_0")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x_orig = x

    x = Conv2D(128, 3, padding="same", name="conv2d_1_1")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(128, 3, padding="same", name="conv2d_1_2")(x)
    x = BatchNormalization()(x)

    x = Add()([x, x_orig])
    x = Activation("relu")(x)

    x = MaxPooling2D((2, 2), (2, 2), name="pool2d_1")(x)

    x = Conv2D(256, 3, padding="same", name="conv2d_2_0")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x_orig = x

    x = Conv2D(256, 3, padding="same", name="conv2d_2_1")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(256, 3, padding="same", name="conv2d_2_2")(x)
    x = BatchNormalization()(x)

    x = Add()([x, x_orig])
    x = Activation("relu")(x)

    x = MaxPooling2D((2, 1), (2, 1), name="pool2d_2")(x)

    x = Conv2D(512, 3, padding="same", name="conv2d_3_0")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x_orig = x

    x = Conv2D(512, 3, padding="same", name="conv2d_3_1")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(512, 3, padding="same", name="conv2d_3_2")(x)
    x = BatchNormalization()(x)

    x = Add()([x, x_orig])

    x = Activation("relu")(x)

    x = MaxPooling2D((2, 1), (2, 1), name="pool2d_3")(x)

    x_orig = x

    x = Conv2D(512, 3, padding="same", name="conv2d_4_1")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(512, 3, padding="same", name="conv2d_4_2")(x)
    x = BatchNormalization()(x)

    x = Add()([x, x_orig])
    x = Activation("relu")(x)

    x = MaxPooling2D((2, 1), (2, 1), name="pool2d_4")(x)

    x = tf.squeeze(x, axis=1)
    # [batch width_seq depth_chanel]

    x = Bidirectional(LSTM(256, return_sequences=True), name="bidirectional_1")(x)
    x = Bidirectional(LSTM(256, return_sequences=True), name="bidirectional_2")(x)
    x = LSTM(512, return_sequences=True)(x)

    x = Dense(len(ALPHABET) + 1)(x)
    y_pred = Activation("softmax", name="softmax")(x)

    Model(inputs=inputs, outputs=y_pred).summary()

    labels = Input(name="the_labels", shape=[None], dtype="float32")
    input_length = Input(name="input_length", shape=[1], dtype="int64")
    label_length = Input(name="label_length", shape=[1], dtype="int64")
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")(
        [y_pred, labels, input_length, label_length]
    )

    sgd = Adam(learning_rate=0.0001,)

    model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={"ctc": lambda y_true, y_pred: y_pred}, optimizer=sgd)

    if weights:
        model.load_weights(weights)
    test_func = K.function([inputs], [y_pred])

    return model, test_func
