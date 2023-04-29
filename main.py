import numpy as np
from numpy import matrix, array
from string import ascii_uppercase as alphabet
import pandas as pd

import hill


def encrypt_test():
    text = 'Attach files by dragging & dropping, selecting or pasting them.'

    # load letter frequencies
    letter_data = pd.read_csv("./english_letters.csv")
    freqs = letter_data['frequency'].tolist()

    key = hill.random_key(5, len(alphabet))

    encrypted = hill.encrypt(text, key, alphabet, freqs=freqs)
    print(f"Encrypted: {encrypted}")

    decrypted = hill.decrypt(encrypted, key, alphabet, freqs=freqs)
    print(f"Decrypted: {decrypted}")


if __name__ == '__main__':
    encrypt_test()
    pass
