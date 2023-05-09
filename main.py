from string import ascii_uppercase as alphabet

import pandas as pd

from crack_cipher import shotgun_hillclimbing, guess_key_len, annealing
from hill_encrypt import encrypt, decrypt
from hill_key import random_key, slide_key
from tests import change_key_performance, perfomence_test, test_shotgun, test_ngram_numbers
from utils import preprocess_text


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
           'building their nests on my boughs, and when the wind blew, I should bow with' \
           '    stately dignity like my tall companions." The tree was so discontented, that it" \
            "took no pleasure in the warm sunshine, the birds, or the rosy clouds that floated' \
           'over it morning and evening. Sometimes, in winter, when the snow lay white and' \
           'glittering on the ground, a hare would come springing along, and jump right over' \
           'the little tree; and then how mortified it would feel!'

    text = text * 4
    processed = preprocess_text(text, alphabet)
    letter_data = pd.read_csv("./english_letters.csv")
    freqs = letter_data['frequency'].tolist()

    key = random_key(key_l, alphabet_len)
    print(f"THE KEY: {key}")

    encrypted = encrypt(processed, key, alphabet, freqs)

    cracked_key, a = shotgun_hillclimbing(encrypted, key_l, alphabet, freqs=freqs, t_limit=60 * 15)
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

def mrie_testing():
    key_l = 2
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
    from crack_cipher import testing
    encrypted = encrypt(processed, key, alphabet, freqs)
    table = testing(encrypted,alphabet,key_l,600, freqs=freqs)
    print(table)
    print(f"THE KEY: {key}")

    pass

if __name__ == '__main__':
    # swap_rows_test()
    # crack_test()

    # guess_me_keys_test()
    # crack_test()
    # # determinant_test()
    # perfomence_test()
    # smart_swap_test()

    # key = random_key(5, 26)
    # changed = add_rows_with_random(key, alphabet_len=26)
    #  Optimizations

    # change_key_performance()

    # crack_test()

    # test_shotgun(alphabet)
    mrie_testing()
    # key = random_key(5, 16)
    # slid = slide_key(key, 26)
    # print(key)
    # print(slid)

    test_ngram_numbers()

    pass
