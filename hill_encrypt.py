import random

from numpy import matrix, ceil, dot

import utils

int_to_char = dict()
char_to_int = dict()
glob_alphabet = []
translation_table = None


def invert_key(matr: matrix, alphabet_len: int):
    """
    Calculate the given key inversion
    :param matr: the key
    :param alphabet_len: the length of the alphabet
    :return: the key inversion
    """
    return utils.mod_inverse_matrix(matr, alphabet_len)


def chunkify(numbers: list[int], chunk_size: int, freqs: list[float] | None = None, alphabet_len: int | None = None) -> \
        list[list[int]]:
    """
    Split list of numbers to chunks of given size.

    If there is any remainder:

        if freqs are provided, select random letter indexes according
        to passed frequencies

        if alphabet_len is provided, select random indexes from 0 to
        alphabet length

        if none of above is provided, select zeros.

    :param numbers: list to split
    :param chunk_size: the size of a single chunk
    :param freqs: optional list of each letter frequency in alphabet
    :param alphabet_len: optional length of alphabet
    :return: the list of numbers splitted into chunks.
    """

    result = []
    n_chunks = int(ceil(len(numbers) / chunk_size))

    for i in range(n_chunks):
        result.append(numbers[i * chunk_size:(i + 1) * chunk_size])

    # if the last chunk is not full, fill it with random letters
    if len(result[-1]) < chunk_size:
        to_draw = chunk_size - len(result[-1])
        if freqs is not None:
            letter_codes = [x for x in range(len(freqs))]
            drawn = random.choices(letter_codes, freqs, k=to_draw)
        elif alphabet_len is not None:
            letter_codes = [x for x in range(alphabet_len)]
            drawn = random.choices(letter_codes, k=to_draw)
        else:
            drawn = [0] * to_draw
        result[-1] = result[-1] + drawn

    return result


def encrypt(text: str, key: matrix, alphabet: str, freqs: list[float] | None = None) -> str:
    """
    :param text: text to encode
    :param key: a key to be used - square matrix
    :param alphabet: the alphabet
    :param freqs: optional; frequencies of each letter in language. If not provided, chunkify function will select random letters
    to fill remainders.
    :return:
    """
    global glob_alphabet, int_to_char, char_to_int
    if alphabet != glob_alphabet:
        glob_alphabet = alphabet
        int_to_char = {k: v for k, v in enumerate(alphabet)}
        char_to_int = {v: k for k, v in enumerate(alphabet)}

    # convert text to list of letter indexes
    alphabet_len = len(alphabet)
    text_numbers = [char_to_int.get(x) for x in text]

    # split text to chunks
    chunks = chunkify(text_numbers, key.shape[0], freqs=freqs, alphabet_len=alphabet_len)

    # v1 <- generally slow
    # encrypted_chunks = []
    # for c in chunks:
    #     matr = matmul(key, c)
    #     encrypted_chunks += matrix(matr % 26).tolist()[0]
    # encrypted_text = "".join([alphabet[x] for x in encrypted_chunks])

    # v2
    # encrypted_chunks = [np.dot(key, c) % alphabet_len for c in chunks] <-fast
    # encrypted_text = ''.join(alphabet[int(x)] for chunk in encrypted_chunks for x in np.ravel(chunk))<- slow

    # v3
    # encrypted_chunks = [(key @ c) % alphabet_len for c in chunks] <-slow
    # encrypted_text = ''.join(alphabet[x] for chunk in encrypted_chunks for x in chunk.flat) <-fast

    # v4
    encrypted_chunks = [dot(key, c) % alphabet_len for c in chunks]
    encrypted_text = ''.join(alphabet[x] for chunk in encrypted_chunks for x in chunk.flat)

    return encrypted_text

def decrypt(text: str, key: matrix, alphabet: str, freqs: list[float] | None = None) -> str:
    """
    The same aproach as in encrypt
    :return: decrypted string
    """
    # calculate inversion
    inv_key = invert_key(key, len(alphabet))

    # return encryption using inverted key (actual decryption)
    return encrypt(text, inv_key, alphabet, freqs)

