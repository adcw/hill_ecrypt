import random
from math import gcd
from numpy import matrix, reshape, linalg, matmul, mod
from utils import are_coprime

import utils


def encrypt():
    """
    how to do:
        n is len of a side of a matrix (height essentially)
        split text to chunks of size n
        convert them to number (can be done before split)
        multiply matrix key with chunk
        where chunk is (  int  )
                       (  int  )
                       (  int  )
        modulo int's in result
        convert back to letters, append them to encrypted text
    :return:
    """
    pass


def encrypt_chunk(key: matrix, chunk: str) -> str:
    """
    encryption of chunk
    :param key: key
    :param chunk: string of size key.shape[0] in open text
    :return: encrpted sting
    """
    text_numbers = [ord(char) - 65 for char in chunk.upper()]
    matr = matmul(key, text_numbers)
    matr = matrix(matr % 26).tolist()[0]

    encrypted = [chr(x + 65) for x in matr]
    return ''.join(encrypted)


def decrypt():
    """
    The same aproach as in encrypt
    :return: decrypted string
    """
    pass


def decrypt_chunk(key: matrix, chunk: str, alphabet_len: int):
    """
    Decryption of a chunk
    :param key: the key used to encrypt the chunk
    :param chunk: the chunk
    :param alphabet_len: the lenght of the alphabet of the chunk
    :return: decrypted chunk
    """

    # calculate key inversion
    inv_key = invert_key(key, alphabet_len)

    # return encrypted chunk
    return encrypt_chunk(inv_key, chunk)


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
    while True:
        key = gen()
        if is_valid_key(key, alphabet_len):
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
