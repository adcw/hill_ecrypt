import numpy as np
from numpy import matrix

import hill
from utils import are_coprime

if __name__ == '__main__':
    alphabet_len = 26
    key_len = 5

    # Generate the key
    key = hill.random_key(key_len, alphabet_len)
    print(key)

    # encypt phrase
    phrase = "aaaab"
    encrypted = hill.encrypt_chunk(key, phrase)
    print(f"{phrase} encrypted: {encrypted}")

    # decrypt phrase
    decrypted = hill.decrypt_chunk(key, encrypted, alphabet_len)
    print(f"{encrypted} decrypted: {decrypted}")
    pass
