import numpy as np
from numpy import matrix

import hill
from utils import are_coprime

if __name__ == '__main__':
    alphabet_len = 26
    key_len = 4

    # Generate coprime integers
    coprime = [x for x in range(alphabet_len) if are_coprime(x, alphabet_len)]

    # Generate the key
    key = hill.random_key(4, coprime)
    print(key)
    a = hill.encrypt_chunk(np.matrix('6 24 1;13 16 10;20 17 15'), "ACT")
    print(a)
    pass
