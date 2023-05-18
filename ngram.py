from math import log10

import numpy as np


class Ngram_score(object):
    def __init__(self, ngram_file, sep=' '):
        """ load a file containing ngrams and counts, calculate log probabilities """
        self.ngrams = {}

        gram_size = None

        for line in open(ngram_file):
            key, count = line.split(sep)

            if gram_size is None:
                gram_size = len(key)

            self.ngrams[key] = int(count)

        self.gram_size = gram_size
        self.val_sum = sum(self.ngrams.values())

        # calculate log probabilities
        for key in self.ngrams.keys():
            self.ngrams[key] = log10(float(self.ngrams[key]) / self.val_sum)

        self.floor = log10(0.01 / self.val_sum)

    def score(self, text):
        """ compute the score of text """
        score = 0
        ngrams = self.ngrams.__getitem__
        for i in range(len(text) - self.gram_size + 1):
            trimmed = text[i:i + self.gram_size]

            if trimmed in self.ngrams:
                score += ngrams(trimmed)
            else:
                score += self.floor
        return score
