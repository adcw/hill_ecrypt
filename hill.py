import random
from numpy import matrix, reshape, linalg, matmul, array, ceil
from utils import are_coprime

import utils


def preprocess_text(text: str, alphabet: str):
    text = text.upper()
    processed = [c for c in text if c in alphabet.upper()]
    return "".join(processed)


def chunkify(numbers: list[int], chunk_size: int, freqs: list[float] | None = None, alphabet_len: int | None = None) -> \
        list[list[int]]:
    """
    Split list of numbers to chunks of given size.

    If there is any reminder:
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

    handle a remainder:
        if there is any remainder left after the split,
        fill if with random letters until it creates a valid chunk.
        Ideally the letters should be chosen according
        to the language's real letter distribution

        for example consider the key of lenght equal to 4 and word:
        STRA WBER Y
        'Y' is a remainder. We should add more letters so it satisfies the key's length:
        for example:
        STRA WBER YADZ
        the next steps are analogous as described above
    :return:
    """

    # preprocess text
    processed = preprocess_text(text, alphabet)

    # convert text to list of letter indexes
    text_numbers = [alphabet.find(x) for x in processed]

    # split text to chunks
    chunks = chunkify(text_numbers, key.shape[0], freqs=freqs, alphabet_len=len(alphabet))

    # encrypt each chunk and join into single string
    encrypted_chunks = array([encrypt_chunk(key, c) for c in chunks]).flatten()

    # convert letter indexes to a string
    encrypted_text = "".join([alphabet[x] for x in encrypted_chunks])

    return encrypted_text


def encrypt_chunk(key: matrix, chunk: list[int]) -> list[int]:
    """
    encryption of chunk
    :param key: key
    :param chunk: a text encoded as indexes of alphabet's letters
    :return: encrypted indexes
    """
    matr = matmul(key, chunk)
    matr = matrix(matr % 26).tolist()[0]

    return matr


def decrypt(text: str, key: matrix, alphabet: str, freqs: list[float] | None = None) -> str:
    """
    The same aproach as in encrypt
    :return: decrypted string
    """
    # calculate inversion
    inv_key = invert_key(key, len(alphabet))

    # return encryption using inverted key (actual decryption)
    return encrypt(text, inv_key, alphabet, freqs)


def decrypt_chunk(key: matrix, chunk: list[int], alphabet_len: int):
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
