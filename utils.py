import os
import random
import sys
import time
from math import gcd
from typing import Callable

from numpy import linalg, matrix, round


# def gcd(a, b):
#     """
#     Oblicza największy wspólny dzielnik dwóch liczb całkowitych.
#     """
#     while b:
#         a, b = b, a % b
#     return a


def disable_print():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enable_print():
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
    text = text.upper()
    processed = [c for c in text if c in alphabet.upper()]
    return "".join(processed)


def quality(callback: Callable, t_: int = 1):
    disable_print()
    t0 = time.time()
    n_iters = 0
    while time.time() - t0 < t_:
        callback()
        n_iters += 1
    enable_print()

    return n_iters


cached_str2ints = None
cached_alphabet = None


def str2ints(text: str, alphabet: str):
    global cached_str2ints, cached_alphabet

    if alphabet != cached_alphabet:
        cached_str2ints = {v: k for k, v in enumerate(alphabet)}
        cached_alphabet = alphabet

    return [cached_str2ints[char] for char in text]
