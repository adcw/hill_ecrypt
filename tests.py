import random
from string import ascii_uppercase as alphabet
from time import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

import hill_encrypt
from hill_encrypt import encrypt
from hill_key import random_key, randomize_rows, smart_rand_rows, is_valid_key, swap_rows, slide_key
from utils import quality, preprocess_text


def perfomence_test():
    """
    generates a report to console about speed of many functions
    """
    # test data
    t_limit: int = 1
    key_len = 4
    alphabet_len = len(alphabet)
    char_to_int = {v: k for k, v in enumerate(alphabet)}
    key = np.matrix([[12, 14, 6, 13], [13, 17, 2, 21], [23, 24, 9, 22], [10, 0, 3, 20]])
    key = random_key(key_len, alphabet_len)
    with open('language_data/english_bigrams.txt', 'r', encoding="UTF-8") as file:
        content = file.readlines()
        splitted = np.array([line.replace("\n", "").split(" ") for line in content])
        splitted[:, 1] = normalize([splitted[:, 1]])
        bigram_data = {k: float(v) for k, v in splitted}
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
    processed = preprocess_text(text, alphabet)
    letter_data = pd.read_csv("language_data/english_letters.csv")
    freqs = letter_data['frequency'].tolist()

    # le test
    print(f"Number of operation per second with key_l = {key_len} and text_len = {len(processed)}:")
    processed = 0
    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        processed = preprocess_text(text, alphabet)
        itr += 1
    print(f'preprocess_text: {itr}')

    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        encrypted = encrypt(processed, key, alphabet, freqs)
        itr += 1
    print(f'encrypt: {itr}')

    from utils import mod_inverse_matrix
    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        mod_inverse_matrix(key, 26)
        itr += 1
    print(f'mod_inverse_matrix: {itr}')

    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        random_key(key_len, alphabet_len)
        itr += 1
    print(f'random_key: {itr}')

    from hill_key import is_valid_key
    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        is_valid_key(key, alphabet_len)
        itr += 1
    print(f'is_valid_key: {itr}')

    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        randomize_rows(key, 0.1, 0.5, alphabet_len)
        itr += 1
    print(f'randomize_rows perc_rows 0.1, perc_elems 0.5: {itr}')

    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        smart_rand_rows(key, processed, alphabet, bigram_data, freqs)
        itr += 1
    print(f'smart_rand_rows: {itr}')

    from hill_encrypt import chunkify
    text_numbers = [char_to_int.get(x) for x in processed]
    itr = 0
    chunks = 0
    t0 = time()
    while time() - t0 < t_limit:
        # split text to chunks
        chunks = chunkify(text_numbers, key.shape[0], freqs=freqs, alphabet_len=alphabet_len)
        itr += 1
    print(f'chunkify (is part of encrypt): {itr}')

    encrypted_chunks = []
    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        # split text to chunks
        encrypted_chunks = [np.dot(key, c) % len(alphabet) for c in chunks]
        itr += 1
    print(f'[np.dot(key, c) % len(alphabet) for c in chunks] (is part of encrypt): {itr}')

    encrypted_chunks = []
    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        # split text to chunks
        encrypted_chunks = [(key @ c) % alphabet_len for c in chunks]
        itr += 1
    print(f'[(key @ c) % alphabet_len for c in chunks]: {itr}')

    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        # split text to chunks
        encrypted_text = ''.join(alphabet[int(x)] for chunk in encrypted_chunks for x in np.ravel(chunk))
        itr += 1
    print(f'"".join(alphabet[int(x)] for chunk in encrypted_chunks for x in np.ravel(chunk)): {itr}')

    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        # split text to chunks
        encrypted_text = ''.join(alphabet[x] for chunk in encrypted_chunks for x in chunk.flat)
        itr += 1
    print(
        f' encrypted_text = "".join(alphabet[x] for chunk in encrypted_chunks for x in chunk.flat) (is part of encrypt): {itr}')


def change_key_performance():
    """
    Generates a report to console about speed of functions that change key
    """

    key_l = 5
    alphabet_len = 26
    key = random_key(key_l, alphabet_len)

    # tests
    is_valid_key_t = quality(lambda: is_valid_key(key, alphabet_len), t_=1)
    print(f"is valid key: {is_valid_key_t}")

    randomize_rows_t = quality(lambda: randomize_rows(key, 0.1, 0.5, alphabet_len))
    print(f"randomize rows: {randomize_rows_t}")

    swap_rows_t = quality(lambda: swap_rows(key))
    print(f"swap rows: {swap_rows_t}")

    slide_key_t = quality(lambda: slide_key(key, alphabet_len))
    print(f"slide_key_t: {slide_key_t}")


def test_inversion():
    """
    test of inversion: can we invert inversion to get original
    """
    sum = 0
    for _ in range(1000):
        original = random_key(5, 26)
        inverted = hill_encrypt.invert_key(original, 26)
        inverted = hill_encrypt.invert_key(inverted, 26)
        if np.array_equal(original, inverted):
            sum += 1

    print(f"Accuray: {sum / 1000:.2f}")


def test_smart_rand():
    """
    Test accuracy: smart part of Function smart_rand_rows
    """
    with open('language_data/english_bigrams.txt', 'r', encoding="UTF-8") as file:
        content = file.readlines()
        splitted = np.array([line.replace("\n", "").split(" ") for line in content])
        splitted[:, 1] = normalize([splitted[:, 1]])
        bigram_data = {k: float(v) for k, v in splitted}
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
    processed = preprocess_text(text, alphabet)
    letter_data = pd.read_csv("language_data/english_letters.csv")
    freqs = letter_data['frequency'].tolist()

    iters = 10000
    count = 0
    key_len = 7
    alphabet_len = 26

    indexes = [x for x in range(key_len)]

    for _ in range(iters):
        real_key = random_key(key_len, alphabet_len)
        real_key_inv = hill_encrypt.invert_key(real_key, alphabet_len)
        encrypted = encrypt(processed, real_key, alphabet, freqs)
        #
        indexes_to_change = random.sample(indexes, k=2)
        real_key_inv_ch = randomize_rows(real_key_inv, 0.01, 0.5, alphabet_len, indexes_to_change)

        _, index_to_change_pred = smart_rand_rows(real_key_inv_ch, encrypted, alphabet, bigram_data, freqs, init=True)

        if index_to_change_pred[0] in indexes_to_change:
            count += 1

    print(f"Accuracy: {count / iters}")

    pass


import random
from string import ascii_uppercase as alphabet
from time import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

import hill_encrypt
from hill_encrypt import encrypt
from hill_key import random_key, randomize_rows, smart_rand_rows, is_valid_key, swap_rows, slide_key
from utils import quality, preprocess_text


def perfomence_test():
    """
    generates a report to console about speed of many functions
    """
    # test data
    t_limit: int = 1
    key_len = 4
    alphabet_len = len(alphabet)
    char_to_int = {v: k for k, v in enumerate(alphabet)}
    key = np.matrix([[12, 14, 6, 13], [13, 17, 2, 21], [23, 24, 9, 22], [10, 0, 3, 20]])
    key = random_key(key_len, alphabet_len)
    with open('language_data/english_bigrams.txt', 'r', encoding="UTF-8") as file:
        content = file.readlines()
        splitted = np.array([line.replace("\n", "").split(" ") for line in content])
        splitted[:, 1] = normalize([splitted[:, 1]])
        bigram_data = {k: float(v) for k, v in splitted}
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
    processed = preprocess_text(text, alphabet)
    letter_data = pd.read_csv("language_data/english_letters.csv")
    freqs = letter_data['frequency'].tolist()

    # le test
    print(f"Number of operation per second with key_l = {key_len} and text_len = {len(processed)}:")
    processed = 0
    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        processed = preprocess_text(text, alphabet)
        itr += 1
    print(f'preprocess_text: {itr}')

    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        encrypted = encrypt(processed, key, alphabet, freqs)
        itr += 1
    print(f'encrypt: {itr}')

    from utils import mod_inverse_matrix
    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        mod_inverse_matrix(key, 26)
        itr += 1
    print(f'mod_inverse_matrix: {itr}')

    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        random_key(key_len, alphabet_len)
        itr += 1
    print(f'random_key: {itr}')

    from hill_key import is_valid_key
    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        is_valid_key(key, alphabet_len)
        itr += 1
    print(f'is_valid_key: {itr}')

    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        randomize_rows(key, 0.1, 0.5, alphabet_len)
        itr += 1
    print(f'randomize_rows perc_rows 0.1, perc_elems 0.5: {itr}')

    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        smart_rand_rows(key, processed, alphabet, bigram_data, freqs)
        itr += 1
    print(f'smart_rand_rows: {itr}')

    from hill_encrypt import chunkify
    text_numbers = [char_to_int.get(x) for x in processed]
    itr = 0
    chunks = 0
    t0 = time()
    while time() - t0 < t_limit:
        # split text to chunks
        chunks = chunkify(text_numbers, key.shape[0], freqs=freqs, alphabet_len=alphabet_len)
        itr += 1
    print(f'chunkify (is part of encrypt): {itr}')

    encrypted_chunks = []
    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        # split text to chunks
        encrypted_chunks = [np.dot(key, c) % len(alphabet) for c in chunks]
        itr += 1
    print(f'[np.dot(key, c) % len(alphabet) for c in chunks] (is part of encrypt): {itr}')

    encrypted_chunks = []
    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        # split text to chunks
        encrypted_chunks = [(key @ c) % alphabet_len for c in chunks]
        itr += 1
    print(f'[(key @ c) % alphabet_len for c in chunks]: {itr}')

    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        # split text to chunks
        encrypted_text = ''.join(alphabet[int(x)] for chunk in encrypted_chunks for x in np.ravel(chunk))
        itr += 1
    print(f'"".join(alphabet[int(x)] for chunk in encrypted_chunks for x in np.ravel(chunk)): {itr}')

    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        # split text to chunks
        encrypted_text = ''.join(alphabet[x] for chunk in encrypted_chunks for x in chunk.flat)
        itr += 1
    print(
        f' encrypted_text = "".join(alphabet[x] for chunk in encrypted_chunks for x in chunk.flat) (is part of encrypt): {itr}')


def change_key_performance():
    """
    Generates a report to console about speed of functions that change key
    """

    key_l = 5
    alphabet_len = 26
    key = random_key(key_l, alphabet_len)

    # tests
    is_valid_key_t = quality(lambda: is_valid_key(key, alphabet_len), t_=1)
    print(f"is valid key: {is_valid_key_t}")

    randomize_rows_t = quality(lambda: randomize_rows(key, 0.1, 0.5, alphabet_len))
    print(f"randomize rows: {randomize_rows_t}")

    swap_rows_t = quality(lambda: swap_rows(key))
    print(f"swap rows: {swap_rows_t}")

    slide_key_t = quality(lambda: slide_key(key, alphabet_len))
    print(f"slide_key_t: {slide_key_t}")


def test_inversion():
    """
    test of inversion: can we invert inversion to get original
    """
    sum = 0
    for _ in range(1000):
        original = random_key(5, 26)
        inverted = hill_encrypt.invert_key(original, 26)
        inverted = hill_encrypt.invert_key(inverted, 26)
        if np.array_equal(original, inverted):
            sum += 1

    print(f"Accuray: {sum / 1000:.2f}")


def test_smart_rand():
    """
    Test accuracy: smart part of Function smart_rand_rows
    """
    with open('language_data/english_bigrams.txt', 'r', encoding="UTF-8") as file:
        content = file.readlines()
        splitted = np.array([line.replace("\n", "").split(" ") for line in content])
        splitted[:, 1] = normalize([splitted[:, 1]])
        bigram_data = {k: float(v) for k, v in splitted}
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
    processed = preprocess_text(text, alphabet)
    letter_data = pd.read_csv("language_data/english_letters.csv")
    freqs = letter_data['frequency'].tolist()

    iters = 10000
    count = 0
    key_len = 7
    alphabet_len = 26

    indexes = [x for x in range(key_len)]

    for _ in range(iters):
        real_key = random_key(key_len, alphabet_len)
        real_key_inv = hill_encrypt.invert_key(real_key, alphabet_len)
        encrypted = encrypt(processed, real_key, alphabet, freqs)
        #
        indexes_to_change = random.sample(indexes, k=2)
        real_key_inv_ch = randomize_rows(real_key_inv, 0.01, 0.5, alphabet_len, indexes_to_change)

        _, index_to_change_pred = smart_rand_rows(real_key_inv_ch, encrypted, alphabet, bigram_data, freqs, init=True)

        if index_to_change_pred[0] in indexes_to_change:
            count += 1

    print(f"Accuracy: {count / iters}")

    pass
