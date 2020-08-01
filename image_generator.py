from trdg.generators import (GeneratorFromDict, GeneratorFromRandom)
import string
import random
import cv2
import time
import numpy as np

class FakeImageGenerator():
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.len_alphabet = len(self.alphabet)
        self.height = 32
        self.channels = 3
        self.batch_size = 16

        self.wiki_generators = [
            GeneratorFromDict(language='fr', background_type=3, fit=True, length=3),
            GeneratorFromRandom(background_type=3, fit=True)
        ]

    def text_to_labels(self, text):
        """ Translation of characters to unique integer values
        """
        ret = []
        for char in text:
            ret.append(self.alphabet.find(char))
        return ret

    def get_image_label(self):
        valid = False
        while valid is False:
            valid = True
            im, label = next(self.current_generator)
            im = np.array(im)
            im = im / 255
            for char in label:
                if char not in self.alphabet:
                    valid = False
                    continue
        return im, self.text_to_labels(label)

    def prepare_batch(self):
        self.current_generator = random.choice(self.wiki_generators)
        #self.current_generator = self.wiki_generators[1]
        images = []
        labels = []
        for i in range(self.batch_size):
            im, text = self.get_image_label()
            images.append(im)
            labels.append(text)
        self.images, self.input_length = self.pad_images_batch(images)
        self.labels, self.label_length = self.pad_labels_batch(labels)

    def pad_images_batch(self, images):
        max_size = max([im.shape[1] for im in images])
        padded_images = []
        # padded_images = np.zeros((self.batch_size, self.height, max_size, self.channels))
        for i, im in enumerate(images):
            im = np.array(im)
            w = im.shape[1]
            im_pad = np.pad(im, ((0, 0), (0, max_size - w), (0, 0)), mode='edge')
            padded_images.append(im_pad)
        return np.array(padded_images), np.ones((self.batch_size,)) * (max_size // 4 - 2)

    def pad_labels_batch(self, labels):
        max_length = max([len(l) for l in labels])
        padded_labels = np.ones((self.batch_size, max_length), dtype="int32") * self.alphabet.find(" ")
        for i, l in enumerate(labels):
            padded_labels[i][0: len(l)] = l
        return padded_labels, np.ones((self.batch_size,)) * max_length

    def gen(self):
        while 1:
            self.prepare_batch()
            inputs = {
                'the_input': self.images,
                'the_labels': self.labels,
                'input_length': self.input_length,
                'label_length': self.label_length,
                }
            outputs = {'ctc': np.zeros([self.batch_size])}  # dummy data for dummy loss function
            yield (inputs, outputs)