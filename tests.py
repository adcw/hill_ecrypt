from string import ascii_uppercase as alphabet
from time import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from hill_encrypt import encrypt, fast_encrypt
from hill_key import random_key, randomize_key, add_rows_with_random, randomize_rows, smart_rand_rows, is_valid_key, \
    swap_rows, slide_key
from utils import disable_print, enable_print, quality, preprocess_text
from crack_cipher import shotgun_hillclimbing

from ngram import NgramNumbers, Ngram_score

from tqdm import tqdm


def perfomence_test():
    # test data
    t_limit: int = 1
    key_len = 4
    alphabet_len = len(alphabet)
    char_to_int = {v: k for k, v in enumerate(alphabet)}
    int_to_char = {k: v for k, v in enumerate(alphabet)}
    key = np.matrix([[12, 14, 6, 13], [13, 17, 2, 21], [23, 24, 9, 22], [10, 0, 3, 20]])
    key = random_key(key_len, alphabet_len)
    with open('./english_bigrams.txt', 'r') as file:
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
    letter_data = pd.read_csv("./english_letters.csv")
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

    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        encrypted = fast_encrypt(processed, key, alphabet, freqs)
        itr += 1
    print(f'fast encrypt: {itr}')

    from utils import mod_inverse_matrix
    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        mod_inverse_matrix(key, 26)
        itr += 1
    print(f'mod_inverse_matrix: {itr}')

    itr = 0
    disable_print()
    t0 = time()
    while time() - t0 < t_limit:
        random_key(key_len, alphabet_len)
        itr += 1
    enable_print()
    print(f'random_key: {itr}')

    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        randomize_key(key, 0.2, alphabet_len)
        itr += 1
    print(f'randomize_key 20%: {itr}')

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
        add_rows_with_random(key, alphabet_len)
        itr += 1
    print(f'add_rows_with_random: {itr}')

    # broken for now
    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        smart_rand_rows(key, processed, alphabet, bigram_data, freqs)
        itr += 1
    print(f'smart_rand_rows: {itr}')

    from hill_encrypt import chunkify, encrypt_chunk
    text_numbers = [char_to_int.get(x) for x in processed]
    itr = 0
    chunks = 0
    t0 = time()
    while time() - t0 < t_limit:
        # split text to chunks
        chunks = chunkify(text_numbers, key.shape[0], freqs=freqs, alphabet_len=alphabet_len)
        itr += 1
    print(f'chunkify (is part of encrypt): {itr}')

    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        # split text to chunks
        encrypt_chunk(key, chunks[0])
        itr += 1
    print(f'encrypt_chunk(key, chunks[0]): {itr}')

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
    disable_print()

    key_l = 5
    alphabet_len = 26
    key = random_key(key_l, alphabet_len)

    # tests
    is_valid_key_t = quality(lambda: is_valid_key(key, alphabet_len), t_=1)
    randomize_rows_t = quality(lambda: randomize_rows(key, 0.5, 0.5, alphabet_len))
    randomize_key_t = quality(lambda: randomize_key(key, 0.5, alphabet_len))
    swap_rows_t = quality(lambda: swap_rows(key))
    add_rows_with_random_t = quality(lambda: add_rows_with_random(key, alphabet_len))
    slide_key_t = quality(lambda: slide_key(key, alphabet_len))
    # smart_rand_rows_t = quality(lambda: smart_rand_rows(key, text, alphabet, bigram_data, freqs))

    enable_print()

    # prints
    print(f"is valid key: {is_valid_key_t}")
    print(f"randomize rows: {randomize_rows_t}")
    print(f"randomize key: {randomize_key_t}")
    print(f"swap rows: {swap_rows_t}")
    print(f"add_rows_with_random_t: {add_rows_with_random_t}")
    print(f"slide_key_t: {slide_key_t}")


def test_shotgun(alphabet_: str, key_len: int = 2, n_tests: int = 5):
    alphabet_len = len(alphabet_)

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

    processed = preprocess_text(text, alphabet_)
    letter_data = pd.read_csv("./english_letters.csv")
    freqs = letter_data['frequency'].tolist()

    real_keys = []
    encryptions = []

    for _ in range(n_tests):
        key = random_key(key_len, alphabet_len)
        encrypted = encrypt(processed, key, alphabet_, freqs)
        real_keys.append(key)
        encryptions.append(encrypted)

    guessed_keys = []
    fitnesses = []

    t0 = time()
    for key, encrypted in tqdm(zip(real_keys, encryptions), total=n_tests):
        disable_print()
        guessed_key, fitness = shotgun_hillclimbing(encrypted, key_len, alphabet_, freqs=freqs)
        enable_print()
        guessed_keys.append(guessed_key)
        fitnesses.append(fitness)
    t = time() - t0

    avg_fitness = np.average(fitnesses)

    n_guessed = 0
    for guessed, real in zip(guessed_keys, real_keys):
        if np.array_equal(guessed, real):
            n_guessed += 1

    effectiveness = n_guessed / n_tests

    print(f"avg time: {t/n_tests:.2f} secs, avg_fitness: {avg_fitness/len(text):.2f}, effectiveness: {effectiveness}")


def test_ngram_numbers():
    scorer = NgramNumbers('./english_bigrams.txt', alphabet)
    sc2 = Ngram_score('./english_bigrams.txt')

    word = 'UNAFLIANLWFNALNWFAFNLAWKMF' * 1000
    word_num = [alphabet.find(x) for x in word]

    print(f"ngrams on text: {sc2.score(word)}")
    print(f"ngrams on numbers: {scorer.score(word_num)}")

    t0 = time()
    for _ in range(100):
        sc2.score(word)
    print(f"ngrams on text time: {time() - t0}")

    t0 = time()
    for _ in range(100):
        scorer.score(word_num)
    print(f"ngrams on numbers time: {time() - t0}")
