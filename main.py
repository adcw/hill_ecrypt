import numpy as np
from numpy import matrix, array
from string import ascii_uppercase as alphabet
import pandas as pd

from hill_encrypt import encrypt, decrypt
import hill_key


def encrypt_test():
    text = 'Attach files by dragging & dropping, selecting or pasting them.'

    # load letter frequencies
    letter_data = pd.read_csv("./english_letters.csv")
    freqs = letter_data['frequency'].tolist()

    key = hill_key.random_key(5, len(alphabet))

    encrypted = encrypt(text, key, alphabet, freqs=freqs)
    print(f"Encrypted: {encrypted}")

    decrypted = decrypt(encrypted, key, alphabet, freqs=freqs)
    print(f"Decrypted: {decrypted}")


def randomize_key_test():
    alphabet_len = len(alphabet)
    key_l = 3
    key = hill_key.random_key(key_l, alphabet_len)
    print(f"Key: \n{key}")

    changed_key = hill_key.randomize_key(key, percentage=0.5, alphabet_len=alphabet_len)
    print(f"Changed key: \n{changed_key}")


def swap_rows_test():
    alphabet_len = len(alphabet)
    key_l = 3
    key = hill_key.random_key(key_l, alphabet_len)
    print(f"Key: \n{key}")

    changed_key = hill_key.swap_rows(key)
    print(f"Changed key: \n{changed_key}")


if __name__ == '__main__':
    swap_rows_test()

    pass
