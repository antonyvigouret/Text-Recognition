import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Bidirectional,
    LSTM,
    Softmax,
    MaxPooling2D,
    BatchNormalization,
    Activation,
    Reshape,
    Input,
    Lambda,
    Add,
)
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_CRNN():
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

    # captures output of softmax so we can decode the output during visualization
    test_func = K.function([input_data], [y_pred])

    return model, test_func

alphabet = "".join(['°', 'Ø', '²']) + "".join([chr(i) for i in range(3, 128)]) + "".join(
            ["é", "è", "à", "û", "ç", "î", "ï"]
)


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

    x = Dense(len(alphabet) + 1)(x)
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

    x = Dense(len(alphabet) + 1)(x)
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
