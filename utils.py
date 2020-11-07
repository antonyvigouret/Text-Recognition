import itertools
import numpy as np

ALPHABET = "".join(['°', 'Ø', '²']) + "".join([chr(i) for i in range(3, 128)]) + "".join(
            ["é", "è", "à", "û", "ç", "î", "ï"]
)


def labels_to_text(labels, alphabet):
    """Reverse translation of numerical classes back to characters."""
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)


def text_to_labels(text, alphabet):
    """ Translation of characters to unique integer values
    """
    ret = []
    for char in text:
        ret.append(alphabet.find(char))
    return ret


def decode_batch(test_func, word_batch, alphabet):
    """
    - Greedy search -
    For a real OCR application, this should be beam
    search with a dictionary and language model.
    For this example, best path is sufficient.
    """
    res = []
    prob = []
    out = test_func([word_batch])[0]
    for i in range(out.shape[0]):
        out_best = list(np.argmax(out[i, 2:], 1))
        out_prob = np.mean(np.max(out[i, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best, alphabet)
        res.append(outstr)
        prob.append(out_prob)
    return res, prob
