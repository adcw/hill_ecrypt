from string import ascii_uppercase as alphabet

import pandas as pd

from crack_cipher import shotgun_hillclimbing, guess_key_len
from hill_encrypt import encrypt, decrypt
from hill_key import random_key
from tests import test_chunkify_text, test_shotgun, change_key_performance, perfomence_test, test_ngram_numbers
from utils import preprocess_text

import winsound


def crack_test():
    key_l = 4
    alphabet_len = len(alphabet)

    with open("./text.txt", "r") as file:
        text = file.read()

    processed = preprocess_text(text, alphabet)
    letter_data = pd.read_csv("./english_letters.csv")
    freqs = letter_data['frequency'].tolist()

    key = random_key(key_l, alphabet_len)
    print(f"THE KEY: {key}")

    encrypted = encrypt(processed, key, alphabet, freqs)

    """
    Best bend values
    key_len | row bend | elem bend
    2       | 1.4      | 1
    3       | 4        | 0.8
    4       | 10       | 0.55
    """

    cracked_key, a = shotgun_hillclimbing(encrypted, key_l, alphabet, freqs=freqs, t_limit=60 * 20,
                                          search_deepness=1000, row_bend=10, elem_bend=0.55)
    cracked_text = decrypt(encrypted, cracked_key, alphabet, freqs)
    print(f"Cracked text: {cracked_text}")

    pass


def guess_me_keys_test():
    key_l = 4
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
           'building their nests on my boughs, and when the wind blew, I should bow with' \
           '    stately dignity like my tall companions." The tree was so discontented, that it" \
            "took no pleasure in the warm sunshine, the birds, or the rosy clouds that floated' \
           'over it morning and evening. Sometimes, in winter, when the snow lay white and' \
           'glittering on the ground, a hare would come springing along, and jump right over' \
           'the little tree; and then how mortified it would feel!'

    letter_data = pd.read_csv("./english_letters.csv")
    freqs = letter_data['frequency'].tolist()
    processed = preprocess_text(text, alphabet)
    key = random_key(key_l, alphabet_len)
    print(f"THE KEY: {key}")

    encrypted = encrypt(processed, key, alphabet, freqs)
    table = guess_key_len(encrypted, alphabet, freqs=freqs)
    print(table)
    print(f'I guess key length is= {table[0][2]}')
    print(f'True key length = {key_l}')

    pass


if __name__ == '__main__':
    # swap_rows_test()
    # crack_test()




    # guess_me_keys_test()
    # crack_test()
    # determinant_test()
    # perfomence_test()
    # smart_swap_test()

    # test_chunkify_text()

    # key = random_key(5, 26)
    # changed = add_rows_with_random(key, alphabet_len=26)
    #  Optimizations

    # change_key_performance()

    crack_test()

    # test_shotgun(alphabet, n_tests=50)
    # mrie_testing()
    # key = random_key(5, 16)
    # slid = slide_key(key, 26)
    # print(key)
    # print(slid)

    # test_ngram_numbers()

    pass
