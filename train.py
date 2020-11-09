import itertools
import os
from time import time

import cv2
import editdistance
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar

from image_generator import FakeImageGenerator
from models import get_CResRNN, get_CRNN, CRNN_model
from train_arg_parser import get_args
from utils import ALPHABET, decode_batch, labels_to_text, text_to_labels
from models import CRNN, ctc_lambda_func


def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best, ALPHABET)
        ret.append(outstr)
    return ret


class VizCallback(tf.keras.callbacks.Callback):
    def __init__(self, run_name, test_func, text_img_gen):
        self.test_func = test_func
        self.output_dir = os.path.join("OUTPUT_DIR", run_name)
        self.text_img_gen = text_img_gen
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def show_edit_distance(self, num):
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        while num_left > 0:
            word_batch = next(self.text_img_gen)[0]
            num_proc = min(word_batch["the_input"].shape[0], num_left)
            decoded_res = decode_batch(
                self.test_func, word_batch["the_input"][0:num_proc]
            )
            for j in range(num_proc):
                pred = decoded_res[j].strip()
                truth = labels_to_text(word_batch["the_labels"][j], ALPHABET)
                edit_dist = editdistance.eval(pred, truth)
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / max(len(truth), len(pred))
            num_left -= num_proc
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        print(
            "\nOut of %d samples:  Mean edit distance: "
            "%.3f / Mean normalized edit distance: %0.3f" % (num, mean_ed, mean_norm_ed)
        )

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(
            os.path.join(self.output_dir, "weights%02d.h5" % (epoch))
        )
        self.show_edit_distance(256)


def train(args):
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x["the_input"])
            # loss = tf.reduce_mean(ctc_lambda_func((y_pred, x["the_labels"], x["input_length"].reshape((-1,1)), x["label_length"].reshape((-1,1)))))
            loss = tf.reduce_mean(ctc_lambda_func((y_pred, x["the_labels"], tf.reshape(x["input_length"], [-1, 1]), tf.reshape(x["label_length"], [-1, 1]))))
        
        # Compute gradients
        trainable_vars = model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        model.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return loss


    epochs = 1000
    iter_per_epoch = 100
    #model, test_func = get_CResRNN(weights=os.path.join("OUTPUT_DIR", "exp1", "weights06.h5"))
    #model, test_func = get_CResRNN(weights=os.path.join("OUTPUT_DIR", "weights0995.h5"))
    #model.load_weights(os.path.join("OUTPUT_DIR", "exp1", "weights15.h5"))
    #model.load_weights(os.path.join("OUTPUT_DIR", "weights0995.h5"))
    model2, test_func = CRNN_model()

    train_generator = FakeImageGenerator(args).next_gen()
    

    model = CRNN(ALPHABET)
    model.build()
    model.summary()

    # model = tf.keras.load_model('checkpoints/checkpoint')
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=5))

    loss_train = []

    for epoch in range(1, epochs):
        print(f"Start of epoch {epoch}")

        pb = Progbar(iter_per_epoch, stateful_metrics="loss")

        for iter in range(iter_per_epoch):
            x, y = next(train_generator)
            with tf.GradientTape() as tape:
                y_pred = model(x["the_input"])
                # loss = tf.reduce_mean(ctc_lambda_func((y_pred, x["the_labels"], x["input_length"].reshape((-1,1)), x["label_length"].reshape((-1,1)))))
                loss = tf.reduce_mean(ctc_lambda_func((y_pred, x["the_labels"], tf.reshape(x["input_length"], [-1, 1]), tf.reshape(x["label_length"], [-1, 1]))))
            
            # Compute gradients
            trainable_vars = model.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            # Update weights
            model.optimizer.apply_gradients(zip(gradients, trainable_vars))

            values = [('loss', loss)]
            pb.add(1, values=values)

        if epoch % 5 == 0:
            model.save("checkpoints/base_crnn.h5")


    
    

    # print("test2")
    # x, y = next(train_generator)
    # model.fit(x, y)
    # print("test1")
    
    x, y = next(train_generator)
    print(model(x["the_input"]))

    """
    model.fit(
        train_generator,
        epochs=1000,
        initial_epoch=0,
        steps_per_epoch=100,
        callbacks=[VizCallback("exp1", test_func, FakeImageGenerator(args).next_gen())],
    )
    """


if __name__ == "__main__":
    args = get_args()
    train(args)
    
    """
    model, test_func = CRNN_model()
    model.load_weights(os.path.join("OUTPUT_DIR", "weights0995.h5"))
    train_generator = FakeImageGenerator(args).next_gen()

    while 1:
        x, y = next(train_generator)
        pred = decode_batch(test_func, x["the_input"])
        for i in range(len(pred)):
            print(pred[i])
            cv2.imshow("im", x["the_input"][i])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    """
