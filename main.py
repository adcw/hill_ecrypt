import numpy as np
from numpy import matrix, array
from string import ascii_uppercase as alphabet
import pandas as pd

from hill_encrypt import encrypt, decrypt
from hill_key import random_key, randomize_key, swap_rows, invert_key
from crack_cipher import shotgun_hillclimbing


def encrypt_test():
    text = 'Attach files by dragging & dropping, selecting or pasting them.'

    # load letter frequencies
    letter_data = pd.read_csv("./english_letters.csv")
    freqs = letter_data['frequency'].tolist()

    key = random_key(5, len(alphabet))

    encrypted = encrypt(text, key, alphabet, freqs=freqs)
    print(f"Encrypted: {encrypted}")

    decrypted = decrypt(encrypted, key, alphabet, freqs=freqs)
    print(f"Decrypted: {decrypted}")
    pass


def randomize_key_test():
    alphabet_len = len(alphabet)
    key_l = 3
    key = random_key(key_l, alphabet_len)
    print(f"Key: \n{key}")

    changed_key = randomize_key(key, percentage=0.5, alphabet_len=alphabet_len)
    print(f"Changed key: \n{changed_key}")


def swap_rows_test():
    alphabet_len = len(alphabet)
    key_l = 3
    key = random_key(key_l, alphabet_len)
    print(f"Key: \n{key}")

    changed_key = swap_rows(key)
    print(f"Changed key: \n{changed_key}")


def crack_test():
    key_l = 3
    alphabet_len = len(alphabet)
    text = 'Since different users might have different needs and different assumptions, the NumPy developers' \
           'refused to guess and instead decided to raise a ValueError whenever one tries to evaluate an array' \
           'in Boolean context. Applying and to two numpy arrays causes the two arrays to be evaluated in Boolean context'

    letter_data = pd.read_csv("./english_letters.csv")
    freqs = letter_data['frequency'].tolist()

    key = random_key(key_l, alphabet_len)
    encrypted = encrypt(text, key, alphabet, freqs)

    cracked_key = shotgun_hillclimbing(encrypted, key_l, alphabet, freqs=freqs)
    cracked_text = decrypt(encrypted, cracked_key, alphabet, freqs)

    pass


if __name__ == '__main__':
    # swap_rows_test()
    # crack_test()

    # key_l = 3
    # alphabet_len = len(alphabet)
    # text = 'Attach files by dragging & dropping, selecting or pasting them.'
    #
    # letter_data = pd.read_csv("./english_letters.csv")
    # freqs = letter_data['frequency'].tolist()
    #
    # key = random_key(key_l, alphabet_len)
    # encrypted = encrypt(text, key, alphabet, freqs)
    # inv_key = invert_key(key, alphabet_len)
    # decrypted = encrypt(text, inv_key, alphabet, freqs)

    crack_test()

    pass
