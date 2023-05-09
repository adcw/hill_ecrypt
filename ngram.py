from math import log10

import numpy as np


class Ngram_score(object):
    def __init__(self, ngramfile, sep=' '):
        """ load a file containing ngrams and counts, calculate log probabilities """
        self.ngrams = {}
        for line in open(ngramfile):
            key, count = line.split(sep)
            self.ngrams[key] = int(count)
        self.L = len(key)
        self.N = sum(self.ngrams.values())

        # calculate log probabilities
        for key in self.ngrams.keys():
            self.ngrams[key] = log10(float(self.ngrams[key]) / self.N)

        self.floor = log10(0.01 / self.N)

    def score(self, text):
        """ compute the score of text """
        score = 0
        ngrams = self.ngrams.__getitem__
        for i in range(len(text) - self.L + 1):
            if text[i:i + self.L] in self.ngrams:
                score += ngrams(text[i:i + self.L])
            else:
                score += self.floor
        return score


class NgramNumbers:
    def __init__(self, filename: str, alphabet: str, sep=' '):
        self.filename = filename
        self.sep = sep
        self.alphabet = alphabet

        self.dict_c2i = {v: k for k, v in enumerate(alphabet)}
        self.dict_i2c = {k: v for k, v in enumerate(alphabet)}

        self.ngrams, self.floor = self._parse_file()

        pass

    def _parse_file(self):
        with open(self.filename) as file:
            d = dict()
            for line in file.readlines():
                key, count = line.split(self.sep)
                count = int(count)
                key = tuple(self.dict_c2i.get(char) for char in key)
                d[key] = count

            self.ngram_len = len(key)

            s = sum(list(d.values()))

            for key in d.keys():
                d[key] = log10(float(d[key]) / s)

            return d, log10(0.01 / s)

    def score(self, text: list[int]):
        score = 0
        ngrams = self.ngrams.__getitem__
        for i in range(len(text) - self.ngram_len + 1):
            k = tuple(text[i: i + self.ngram_len])
            if k in self.ngrams:
                score += ngrams(k)
            else:
                score += self.floor

        return score
