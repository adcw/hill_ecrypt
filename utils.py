import operator
import os
import sys
import time
from math import gcd
from typing import Callable

from numpy import linalg, matrix, round
from pandas import read_csv


def disable_print():
    """
    This function sends all future prints into a null file
    """
    sys.stdout = open(os.devnull, 'w')


# Restore
def enable_print():
    """
    This function sends all future prints into console
    """
    sys.stdout = sys.__stdout__


def are_coprime(a, b):
    """
    Checks if two values are coprime.
    """
    return gcd(a, b) == 1


def mod_inverse_matrix(m: matrix, modulo: int) -> matrix | None:
    """
    Modulo inverse of an matrix
    :param m: the matrix
    :param modulo: the modulo
    :return: a matrix inversion
    """
    det = round(linalg.det(m))

    if gcd(int(det), modulo) != 1:
        return None

    m_inv = linalg.inv(m)
    m_inv_modulo = (m_inv * det * pow(int(det), -1, modulo)) % modulo
    m_int = round(m_inv_modulo).astype(int) % modulo
    return m_int


def preprocess_text(text: str, alphabet: str):
    """
    :param text: raw text
    :param alphabet: alphabet used in text (lower or upper)
    :return: text that contain only letters
    """
    text = text.upper()
    processed = [c for c in text if c in alphabet.upper()]
    return "".join(processed)


def quality(callback: Callable, t_: int = 1):
    """
    :param callback: Callable function
    :param t_: test time
    :return: numer of iteration in test time
    """
    t0 = time.time()
    n_iters = 0
    while time.time() - t0 < t_:
        callback()
        n_iters += 1

    return n_iters


def generate_grams(in_file_path: str, out_file_path: str, alphabet: str, n: int = 2):
    with open(in_file_path, encoding='UTF-8') as file_in:
        text = file_in.read()
        text = preprocess_text(text, alphabet)

    dictionary = dict()

    for i in range(0, len(text) - n):
        gram = text[i:i + n]

        if gram in dictionary:
            dictionary[gram] += 1
        else:
            dictionary[gram] = 1

    entries = [f"{key} {value}" for key, value in
               sorted([i for i in dictionary.items()], key=operator.itemgetter(1), reverse=True)]

    text = "\n".join(entries)

    with open(out_file_path, encoding='UTF-8', mode='w+') as file:
        file.write(text)


def get_german():
    german_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜß"
    german_bigrams = "./german_bigrams.txt"
    german_trigrams = "./german_trigrams.txt"
    german_letter_freqs = "./german_letters.csv"

    letter_data = read_csv(german_letter_freqs)
    freqs = letter_data['frequency'].tolist()

    return german_alphabet, german_bigrams, german_trigrams, freqs
