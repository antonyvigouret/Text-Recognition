import os
from glob import glob
import numpy as np
import random
from time import time

import cv2

import tensorflow as tf

from utils import ALPHABET

from trdg.generators import (
    GeneratorFromWikipedia,
    GeneratorFromRandom,
    GeneratorFromDict,
)

from albumentations import (
    IAAPerspective,
    ShiftScaleRotate,
    Transpose,
    Blur,
    OpticalDistortion,
    HueSaturationValue,
    IAAAdditiveGaussianNoise,
    GaussNoise,
    MotionBlur,
    MedianBlur,
    IAAPiecewiseAffine,
    IAASharpen,
    IAAEmboss,
    RandomBrightnessContrast,
    OneOf,
    Compose,
    RandomCrop,
    ElasticTransform,
    RandomGamma,
)


class FakeImageGenerator(tf.keras.utils.Sequence):
    """
    This object is used to handle the image dataset.
    """

    def __init__(self, args, training=True, semi_amount=0.0):
        """
        Initialize the constructor.
        """
        self.args = args
        self.is_training = training
        self.batch_size = args.batch_size
        self.current_iter = 0

        self.fake_generators = [
            GeneratorFromRandom(
                count=-1,
                length=1,
                language="fr",
                size=64,
                background_type=1,
                skewing_angle=2,
                margins=(2, 1, 1, 1),
                use_letters=False,
                use_symbols=False,
                use_numbers=True,
                random_skew=True,
                text_color='#000000,#888888',
            ),
            GeneratorFromRandom(
                count=-1,
                length=1,
                language="fr",
                size=48,
                background_type=1,
                skewing_angle=2,
                margins=(2, 1, 1, 1),
                use_letters=True,
                use_symbols=True,  # false
                use_numbers=False,
                random_skew=True,
                text_color='#000000,#888888',
            ),
            GeneratorFromRandom(
                count=-1,
                length=3,
                language="fr",
                size=24,
                background_type=3,
                skewing_angle=2,
                fit=True,
                random_skew=True,
                text_color='#000000,#888888',
            ),
            GeneratorFromRandom(
                count=-1,
                length=3,
                language="fr",
                size=32,
                background_type=3,
                skewing_angle=2,
                space_width=2,
                use_symbols=False,
                margins=(8, 8, 8, 8),
                random_skew=False,
            ),
            GeneratorFromRandom(
                count=-1,
                length=2,
                language="fr",
                size=55,
                background_type=3,
                skewing_angle=3,
                use_symbols=True,  # false
                fit=True,
                random_skew=True,
                text_color='#0171ff',
            ),
            GeneratorFromRandom(
                count=-1,
                length=3,
                language="fr",
                size=43,
                background_type=1,
                skewing_angle=2,
                margins=(4, 2, 10, 6),
                random_skew=False,
                text_color='#000000,#888888',
            ),
            GeneratorFromRandom(
                count=-1,
                length=5,
                language="fr",
                size=37,
                space_width=3,
                background_type=1,
                use_symbols=False,
                fit=True,
                text_color='#000000,#888888',
            ),
            GeneratorFromRandom(
                count=-1,
                length=5,
                language="fr",
                size=28,
                background_type=3,
                use_symbols=False,
                fit=True,
                text_color='#000000,#888888',
            ),
        ]
        self.wiki_generators = [
            GeneratorFromWikipedia(
                count=-1, language="fr", background_type=3, fit=True
            ),
            GeneratorFromWikipedia(
                count=-1, language="fr", background_type=1, fit=True
            ),
            GeneratorFromWikipedia(
                count=-1, language="fr", background_type=0, fit=True
            ),
            GeneratorFromWikipedia(
                count=-1, language="fr", background_type=3, margins=(5, 2, 7, 1)
            ),
        ]
        self.dict_generators = [
            GeneratorFromDict(
                length=5,
                allow_variable=True,
                language="fr",
                size=32,
                background_type=3,
                fit=True,
                text_color='#000000,#888888',
            ),
            GeneratorFromDict(
                length=5,
                allow_variable=True,
                language="fr",
                size=32,
                background_type=3,
                margins=(7, 4, 6, 4),
                text_color='#000000,#888888',
            ),
            GeneratorFromDict(
                length=5,
                allow_variable=True,
                language="fr",
                size=32,
                background_type=1,
                fit=True,
                text_color='#000000,#888888',
            ),
        ]
        self.classic_gen = [
            GeneratorFromDict(
                length=3,
                allow_variable=True,
                language="fr",
                size=32,
                background_type=1,
                fit=True,
            ),
            GeneratorFromRandom(
                count=-1,
                length=5,
                language="fr",
                size=28,
                background_type=1,
                use_symbols=False,
                fit=True,
            ),
        ]
        self.height = 32
        self.width = None

        self.alphabet = ALPHABET
        # self.alphabet = string.printable
        self.alphabet_size = len(self.alphabet)

    def text_to_labels(self, text):
        """Translation of characters to unique integer values"""
        ret = []
        for char in text:
            if self.alphabet.find(char) >= self.alphabet_size:
                print(text)
            if self.alphabet.find(char) == -1 or self.alphabet.find(char) == 136:
                print(text)
                ret.append(32)
            else:
                ret.append(self.alphabet.find(char))
        return ret

    def get_fake_image_gt(self):
        valid_text = False
        w = 1000
        while w > 500 or valid_text is False:
            im, text = next(self.current_generator)
            valid_text = True
            for char in text:
                if char not in self.alphabet:
                    valid_text = False
                    continue
            im = np.array(im)
            w = im.shape[1]
        im = np.array(im)
        return im, text

    def pad_to_largest_image(self, images):
        sizes = [im.shape[1] for im in images]
        max_size = max(sizes)
        input_length = np.array([max_size // 4 - 2 for im in images])
        images = [
            np.pad(im, ((0, 0), (0, max_size - im.shape[1]), (0, 0)), mode="edge")
            for im in images
        ]
        return images, input_length

    def pad_to_largest_label(self, labels):
        label_length = np.array([len(s) for s in labels])
        maxlen = max(len(s) for s in labels)
        labels_array = np.ones([self.batch_size, maxlen], dtype=int) * 32
        for i in range(self.batch_size):
            for j in range(len(labels[i])):
                labels_array[i][j] = int(labels[i][j])
        return labels_array, label_length

    def prepare_batch(self):
        # chose a generator to generate the whole batch
        r = random.random()
        if r < 1:
            self.current_generator = random.choice(self.classic_gen)
        elif r < 0.75:
            self.current_generator = random.choice(self.fake_generators)
        else:
            self.current_generator = random.choice(self.dict_generators)
        images = []
        labels = []
        source_strings = []
        for _ in range(self.batch_size):
            w = 0
            while w < 10:
                im, text = self.get_fake_image_gt()
                w = im.shape[1]
            im = im / 255
            try:
                im = self.augment_image(im.astype("float32"))
            except:
                pass
            if im.shape[1] < 5 * len(text) + 2:
                new_w = int(5 * len(text))
            else:
                new_w = im.shape[1] + random.randint(4, 16)
            im = cv2.resize(im, (new_w, self.height))
            label = self.text_to_labels(text)
            images.append(im)
            labels.append(label)
            source_strings.append(text)
        images, input_length = self.pad_to_largest_image(images)
        labels, label_length = self.pad_to_largest_label(labels)
        self.batch = np.array(images)
        self.labels = labels
        self.input_length = input_length
        self.label_length = label_length
        self.source_strings = source_strings

    def get_iter(self):
        return self.current_iter

    def next(self):
        self.current_iter += 1
        self.prepare_batch()
        inputs = {
            "the_input": self.batch,
            "the_labels": self.labels,
            "input_length": self.input_length,
            "label_length": self.label_length,
            "source_strings": self.source_strings,
        }
        outputs = {
            "ctc": np.zeros([self.batch_size])
        }  # dummy data for dummy loss function
        return (inputs, outputs)

    def next_gen(self):
        while 1:
            self.current_iter += 1
            self.prepare_batch()
            inputs = {
                "the_input": self.batch,
                "the_labels": self.labels,
                "input_length": self.input_length,
                "label_length": self.label_length,
                #'source_strings': self.source_strings
            }
            outputs = {
                "ctc": np.zeros([self.batch_size])
            }  # dummy data for dummy loss function
            yield (inputs, outputs)

    def augment_image(self, image):
        def strong_aug(p=0.5):
            return Compose(
                [
                    ShiftScaleRotate(
                        shift_limit=0.0125, scale_limit=0.03, rotate_limit=0.5, p=0.8
                    ),
                    RandomGamma(gamma_limit=(80, 120)),
                    RandomBrightnessContrast(),
                    OneOf(
                        [
                            MotionBlur(p=0.6),
                            MedianBlur(blur_limit=3, p=0.5),
                            MedianBlur(blur_limit=(5, 7), p=0.3),
                            Blur(blur_limit=3, p=0.5),
                            Blur(blur_limit=(5, 7), p=0.3),
                        ],
                        p=0.6,
                    ),
                    RandomCrop(image.shape[0] - 10, image.shape[1] - 6, p=0.6),
                    OneOf(
                        [
                            OpticalDistortion(),
                        ],
                        p=0.8,
                    ),
                ],
                p=p,
            )

        augmentation = strong_aug(p=1)
        data = {"image": image}
        augmented = augmentation(**data)
        im_aug = augmented["image"]
        # im_aug = cv2.cvtColor(im_aug, cv2.COLOR_BGR2GRAY)
        return im_aug


class ImageGenerator:
    """
    This object is used to handle the image dataset.
    """

    def __init__(self, args, training=True, semi_amount=0.0):
        """
        Initialize the constructor.
        """
        self.args = args
        self.is_training = training
        self.batch_size = args.batch_size

        # Setting paths
        self.set_folder = args.train_folder if self.is_training else args.val_folder
        self.set_folder = os.path.join(self.args.dataset, self.set_folder)
        self.image_search_path = os.path.join(
            self.set_folder, self.args.image_folder, "*." + self.args.image_ext
        )
        self.label_search_path = os.path.join(
            self.set_folder, self.args.label_folder, "*." + self.args.label_ext
        )
        print(self.image_search_path)
        self.image_paths = glob(self.image_search_path, recursive=True)
        random.shuffle(self.image_paths)
        assert len(self.image_paths) > 0, "Error, no images found in {}".format(
            self.image_search_path
        )
        self.label_paths = {
            os.path.basename(path)[:-7]: path for path in glob(self.label_search_path)
        }
        # assert len(self.label_paths) > 0 and len(self.label_paths) == len(self.image_paths), "Error, missing labels in {}. Got {}# labels".format(self.label_search_path, len(self.label_paths))

        self.current_index = 0

        # Training parameters
        self.max_index = len(self.image_paths)
        self.height = 32

    def reset(self):
        self.current_index = 0

    def __len__(self):
        """
        Return the number of images.
        """
        return int(float(len(self.image_paths)) / float(self.batch_size))

    def get_steps(self):
        return self.__len__()

    def get_image_gt(self, im_path, gt_path):
        x = cv2.imread(im_path)
        fic = open(gt_path, "r")
        label = fic.readline().strip()
        return x, label

    def preprocess_image(self, image):
        x = image / 255
        h, w, c = x.shape
        scale = w / h
        new_h = 32
        new_w = int(scale * new_h)
        try:
            x = cv2.resize(x, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            print(e)
        return x

    def next(self):
        """
        Return the next image. When arriving at the end, shuffle then start again.
        """
        images = []
        gts = []

        for i in range(0, self.batch_size):
            if self.current_index > self.max_index - 1:
                self.current_index = 0
                random.shuffle(self.image_paths)

            # Get current images
            image_path = self.image_paths[self.current_index]
            gt_path = self.label_paths[os.path.basename(image_path)[:-4]]
            image, gt = self.get_image_gt(image_path, gt_path)
            image = self.preprocess_image(image)

            images.append(image)
            gts.append(gt)

            # Increment index variable
            self.current_index += 1
        # print(np.array(gts))
        # print(tf.keras.utils.to_categorical(gts, num_classes=4))
        return (
            np.array(images),
            np.array(gts),
        )  # tf.keras.utils.to_categorical(gts, num_classes=4)

    def next_gen(self):
        """
        Generate one pair of image and groundtruth text
        """
        while 1:
            if self.current_index > self.max_index - 1:
                self.current_index = 0
                random.shuffle(self.image_paths)

            # Get current images
            image_path = self.image_paths[self.current_index]
            gt_path = self.label_paths[os.path.basename(image_path)[:-4]]
            image = None
            while image is None:
                image, gt = self.get_image_gt(image_path, gt_path)
            image = self.preprocess_image(image)
            if image is None:
                print(image_path)
            # Increment index variable
            self.current_index += 1
            yield np.array(image), gt

    def augment_image(self, image):
        def strong_aug(p=0.5):
            return Compose(
                [
                    ShiftScaleRotate(
                        shift_limit=0.0125, scale_limit=0.03, rotate_limit=0.5, p=0.5
                    ),
                    OneOf(
                        [
                            MotionBlur(p=0.3),
                            # MedianBlur(blur_limit=3, p=0.3),
                            Blur(blur_limit=3, p=0.3),
                        ],
                        p=0.4,
                    ),
                    RandomCrop(30, image.shape[1] - 2),
                ],
                p=p,
            )

        augmentation = strong_aug(p=1)
        data = {"image": image}
        augmented = augmentation(**data)
        im_aug = augmented["image"]
        # im_aug = cv2.cvtColor(im_aug, cv2.COLOR_BGR2GRAY)
        return im_aug
