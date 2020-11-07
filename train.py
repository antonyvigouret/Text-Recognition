import os
import string
import numpy as np
import cv2
import itertools
import editdistance
import tensorflow as tf

from image_generator import FakeImageGenerator
from models import get_CRNN, get_CResRNN

from image_generator import FakeImageGenerator
from train_arg_parser import get_args
from utils import decode_batch, labels_to_text

alphabet = "".join(['°', 'Ø', '²']) + "".join([chr(i) for i in range(3, 128)]) + "".join(
            ["é", "è", "à", "û", "ç", "î", "ï"]
)


def text_to_labels(text):
    """ Translation of characters to unique integer values
    """
    ret = []
    for char in text:
        ret.append(alphabet.find(char))
    return ret


def labels_to_text(labels):
    """ Reverse translation of numerical classes back to characters
    """
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)


def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
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
                truth = labels_to_text(word_batch["the_labels"][j])
                edit_dist = editdistance.eval(pred, truth)
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / max(len(truth), len(pred))
            num_left -= num_proc
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        print(
            "\nOut of %d samples:  Mean edit distance:"
            "%.3f Mean normalized edit distance: %0.3f" % (num, mean_ed, mean_norm_ed)
        )

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(
            os.path.join(self.output_dir, "weights%02d.h5" % (epoch))
        )
        self.show_edit_distance(256)


def train(args):
    model, test_func = get_CResRNN(weights=os.path.join("OUTPUT_DIR", "exp1", "weights06.h5"))
    #model.load_weights(os.path.join("OUTPUT_DIR", "exp1", "weights15.h5"))
    train_generator = FakeImageGenerator(args).next_gen()

    model.fit_generator(
        train_generator,
        epochs=1000,
        initial_epoch=5,
        steps_per_epoch=100,
        callbacks=[VizCallback("exp1", test_func, FakeImageGenerator(args).next_gen())],
    )


if __name__ == "__main__":
    args = get_args()
    train(args)
    

    model, test_func = get_CRNN()
    model.load_weights(os.path.join("OUTPUT_DIR", "exp1", "weights19.h5"))
    train_generator = FakeImageGenerator(alphabet).gen()

    while 1:
        x, y = next(train_generator)
        pred = decode_batch(test_func, x["the_input"])
        for i in range(len(pred)):
            print(pred[i])
            cv2.imshow("im", x["the_input"][i])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
