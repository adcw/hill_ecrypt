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


def randomize_rows(key: matrix, perc_rows: float, perc_elems: float, alphabet_len: int):
    def randomize():
        key_len = key.shape[0]
        n_rows_to_change = int(ceil(key_len * perc_rows))
        indexes = [x for x in range(key_len)]
        chosen_rows = random.sample(indexes, n_rows_to_change)

        changed = key.copy()

        for row_index in chosen_rows:
            row = key[row_index].copy().tolist()[0]
            n_elems_to_change = int(ceil(key_len * perc_elems))
            chosen_elems = random.sample(indexes, n_elems_to_change)

            for elem_index in chosen_elems:
                row[elem_index] = (row[elem_index] + random.randint(0, alphabet_len - 1)) % alphabet_len

            changed[row_index] = row

        return changed

    # repeat until a valid key is generated
    while True:
        randomized_key = randomize()
        if is_valid_key(randomized_key, alphabet_len):
            return randomized_key


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
            m_list[i] = (m_list[i] + random.randint(1, alphabet_len - 1)) % alphabet_len

        # return matrix with list's elements
        return matrix(reshape(m_list, (len(key), len(key))))

    # repeat until a valid key is generated
    while True:
        flipped_key = randomize()
        if is_valid_key(flipped_key, alphabet_len):
            return flipped_key


def swap_rows(key: matrix) -> matrix:
    """
    Swap two random rows from a key.
    The operation doesn't require a key validation,
    because swapping a square matrix rows doesn't change it's
    determinant at all.
    :param key: the key
    :return: the key with swapped rows
    """

    # the length of a key
    key_len = key.shape[0]

    # the list of row indexes
    indexes = [x for x in range(key_len)]

    # choose two random indexes
    iloc1, iloc2 = random.sample(indexes, k=2)

    # swap rows of giver indexes
    temp = key[iloc1].copy()
    key[iloc1] = key[iloc2]
    key[iloc2] = temp

    # return swapped key, we don't have to check if the key is still valid,
    # because swapping two rows of a square matrix doesn't change it's determinant.
    return key

def add_rows_with_random(key: matrix, alphabet_len: int) -> matrix:
    x = random.randint(1, alphabet_len-1)

    indexes = [x for x in range(key.shape[0])]
    iloc1, iloc2 = random.sample(indexes, 2)
    # Get the row to multiply
    row_to_multiply = key[iloc1]
    row_to_change = key[iloc2]

    # Multiply the row by the scalar
    new_row = row_to_multiply * x + row_to_change
    new_row = new_row % 26
    new_row = new_row.tolist()[0]
    # Replace the old row with the new one in the matrix
    key[iloc2] = new_row
    return key
