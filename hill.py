import random

from numpy import matrix, reshape, linalg, matmul, mod

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
def encrypt_chunk(key: matrix,chunk: str) -> str:
    """
    encryption of chunk
    :param key: key
    :param chunk: string of size key.shape[0] in open text
    :return: encrpted sting
    """
    text_numbers = [ord(char) - 65 for char in chunk.upper()]
    matr = matmul(key,text_numbers)
    matr = matrix(matr % 26).tolist()[0]

    encrypted = [chr(x + 65) for x in matr]
    return ''.join(encrypted)

def decrypt():
    pass


def random_key(key_len: int, key_elems: list[int]):
    """
    Generate random key from given array of ints

    :param key_len: The length of key (the dimension of square matrix)
    :param key_elems: Elements used to build key
    Must be co-prime with the alphabet length for key to work
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
