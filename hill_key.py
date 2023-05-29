import random

import numpy as np
from numpy import matrix, reshape, ceil
from numpy.linalg import linalg

from hill_encrypt import encrypt
from utils import are_coprime


def random_key(key_len: int, alphabet_len: int):
    """
    Generate random key from given array of ints

    :param alphabet_len: The length of the alphabet
    :param key_len: The length of key (the dimension of square matrix)
    :param key_elems: Elements used to build key
    Must be co-prime with the alphabet length for key to work
    :return: The key as matrix
    """

    # generate array of letter indexes
    key_elems = [i for i in range(alphabet_len)]

    # generating random key
    def gen():
        # Choose random elements to build the matrix
        elems = random.choices(key_elems, k=key_len ** 2)

        # Return array reshaped to a matrix
        return matrix(reshape(elems, (key_len, key_len)))

    # repeat until the key is valid,
    # then return valid key
    iters = 0
    while True:
        key = gen()
        iters += 1
        if is_valid_key(key, alphabet_len):
            print(f"Key generated in {iters} iterations")
            return key


def is_valid_key(key: matrix, alphabet_len: int):
    """
    Checks if a square matrix is a valid key in given alphabet length.
    :param key: the key
    :param alphabet_len: the alphabet's length
    :return: boolean, decision if the key is valid
    """

    # calculate the determinant
    det = round(linalg.det(key))

    # the key is valid if and only if the determinant is non-zero
    # and is coprime with alphabet length
    return det != 0 and are_coprime(det, alphabet_len)


def randomize_rows(key: matrix, perc_rows: float, perc_elems: float, alphabet_len: int,
                   r_indexes: list[int] | None = None, n_rows: int | None = None):
    key_len = key.shape[0]
    n_rows_to_change = n_rows if n_rows is not None else int(ceil(key_len * perc_rows))
    n_elems_to_change = int(ceil(key_len * perc_elems))
    indexes = [x for x in range(key_len)]
    chosen_rows = random.sample(indexes, n_rows_to_change) if r_indexes is None else r_indexes
    chosen_elems = random.sample(indexes, n_elems_to_change)

    def randomize(k):
        for row_index in chosen_rows:
            row = key[row_index].tolist()[0]

            for elem_index in chosen_elems:
                row[elem_index] = (row[elem_index] + random.randint(0, alphabet_len - 1)) % alphabet_len

            k[row_index] = row

        return k

    # repeat until a valid key is generated
    while True:
        temp = key.copy()
        randomized_key = randomize(temp)
        if is_valid_key(randomized_key, alphabet_len):
            return randomized_key


def swap_rows(key: matrix) -> matrix:
    """
    Swap two random rows from a key.
    The operation doesn't require a key validation,
    because swapping a square matrix rows doesn't change it's
    determinant at all.
    :param key: the key
    :return: the key with swapped rows
    """
    copied = key.copy()

    # the length of a key
    key_len = copied.shape[0]

    # the list of row indexes
    indexes = [x for x in range(key_len)]

    # choose two random indexes
    iloc1, iloc2 = random.sample(indexes, k=2)

    # swap rows of giver indexes
    temp = copied[iloc1].copy()
    copied[iloc1] = copied[iloc2]
    copied[iloc2] = temp

    # return swapped key, we don't have to check if the key is still valid,
    # because swapping two rows of a square matrix doesn't change it's determinant.
    return copied


def slide_key(key, alphabet_len: int, horizontal: bool = False) -> matrix:
    temp = key.copy()

    def slide(k):
        l = key.shape[0]
        loc = random.randint(0, l - 1)

        if not horizontal:
            k = k.T

        row = k[loc].tolist()[0]
        slid = [row[(i + 1) % l] for i in range(l)]
        k[loc] = slid

        return k.T if not horizontal else k

    while True:
        result = slide(temp)
        if is_valid_key(result, alphabet_len):
            return result

cached_to_change: list[int] | None = None


def smart_rand_rows(key: matrix, cipher: str, alphabet: str, bigram_data: dict, freqs: list[float] | None = None,
                    init: bool = False, perc_rows: float = 1, n_rows: int | None = None):
    global cached_to_change
    alphabet_len = len(alphabet)

    if cached_to_change is None or init:
        key_len = key.shape[0]
        decrypted_err = encrypt(cipher, key, alphabet, freqs)

        bigram_values = []
        for i in range(len(decrypted_err) - 1):
            chars = decrypted_err[i:i + 2]
            bd = bigram_data.get(chars)

            if bd is None:
                bd = 1

            bigram_values.append(bd)

        recalculated = [2 * bigram_values[0]]
        for i in range(len(bigram_values) - 1):
            x, y = bigram_values[i: i + 2]
            recalculated.append(x + y)

        recalculated.append(2 * bigram_values[-1])
        recalculated = reshape(recalculated, (int(len(recalculated) / key_len), key_len))

        summed = recalculated.sum(axis=0)

        cached_to_change = np.argmax([summed], axis=1).tolist()

    fixed = randomize_rows(key, r_indexes=cached_to_change, alphabet_len=alphabet_len,
                           perc_elems=perc_rows,
                           perc_rows=0.01, n_rows=n_rows)
    return fixed, cached_to_change
