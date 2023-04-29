import random
from numpy import matrix, reshape, ceil
from numpy.linalg import linalg

import utils
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


def invert_key(matr: matrix, alphabet_len: int):
    """
    Calculate the given key inversion
    :param matr: the key
    :param alphabet_len: the length of the alphabet
    :return: the key inversion
    """
    return utils.mod_inverse_matrix(matr, alphabet_len)


def change_key():
    pass


def randomize_key(key: matrix, percentage: float, alphabet_len: int) -> matrix:
    """
    Randomizes elements of a key by choosing random positions of the matrix and
    adding random value in range(1, alphabet_len - 1).
    Each changed element is completely different from the base element.

    :param alphabet_len: the length of the alphabet
    :param key: key to change
    :param percentage: the percentage of key to change. The number of changed elements is always in range from 1 to number of elements in matrix
    :return: changed key
    """

    # randomize some elements of matrix
    def randomize():

        # convert matrix to a list for easier access
        m_list = reshape(key, (len(key) ** 2)).tolist()[0]
        m_len = len(m_list)

        # calculate number of elements to change
        to_change = int(ceil(m_len * percentage))

        # generate a list of indexes
        indexes = [x for x in range(m_len)]

        # chose random indexes from index list
        indexes_to_change = random.sample(indexes, k=to_change)

        # iterate over indexes and change the list accordingly
        for i in indexes_to_change:
            m_list[i] = (m_list[i] + random.randint(1, 25)) % alphabet_len

        # return matrix with list's elements
        return matrix(reshape(m_list, (len(key), len(key))))

    # repeat until a valid key is generated
    while True:
        flipped_key = randomize()
        if is_valid_key(flipped_key, alphabet_len):
            return flipped_key
