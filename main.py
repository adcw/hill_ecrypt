import numpy as np
from numpy import matrix, array
from string import ascii_uppercase as alphabet
import pandas as pd

from hill_encrypt import encrypt, decrypt
from hill_key import random_key, randomize_key, swap_rows, invert_key, randomize_rows
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
    text = 'Far down in the forest, where the warm sun and the fresh air made a sweet' \
           'resting-place, grew a pretty little fir-tree; and yet it was not happy, it wished so' \
           'much to be tall like its companions—the pines and firs which grew around it.' \
           'The sun shone, and the soft air fluttered its leaves, and the little peasant children' \
           'passed by, prattling merrily, but the fir-tree heeded them not. Sometimes the' \
           'children would bring a large basket of raspberries or strawberries, wreathed on a' \
           'straw, and seat themselves near the fir-tree, and say, "Is it not a pretty little tree?"' \
           'which made it feel more unhappy than before. And yet all this while the tree' \
           'grew a notch or joint taller every year; for by the number of joints in the stem of' \
           'a fir-tree we can discover its age. Still, as it grew, it complained, "Oh! how I" \
           "wish I were as tall as the other trees, then I would spread out my branches on' \
           'every side, and my top would over-look the wide world. I should have the birds' \
           'zbuilding their nests on my boughs, and when the wind blew, I should bow with' \
           '    stately dignity like my tall companions." The tree was so discontented, that it" \
            "took no pleasure in the warm sunshine, the birds, or the rosy clouds that floated' \
           'over it morning and evening. Sometimes, in winter, when the snow lay white and' \
           'glittering on the ground, a hare would come springing along, and jump right over' \
           'the little tree; and then how mortified it would feel!'

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

    # key = random_key(5, 26)
    # changed = randomize_rows(key, perc_rows=0.1, perc_elems=0.2, alphabet_len=26)

    pass
