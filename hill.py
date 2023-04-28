import random

from numpy import matrix, reshape, linalg

def encrypt():
    pass


def decrypt():
    pass


def random_key(key_len: int, key_elems: list[int]):
    """
    Generate random key from given array of ints

    :param key_len: The length of key (the dimension of square matrix)
    :param key_elems: Elements used to build key
    Must be co-prime with the alphabet for key to work
    :return: The key as matrix
    """

    # Chose random elements to build the matrix
    elems = random.choices(key_elems, k=key_len ** 2)

    # reshape the list of elements to a square matrix and return
    return matrix(reshape(elems, (key_len, key_len)))


def invert_key(matr: matrix):
    return linalg.inv(matr)


def change_key():
    pass
