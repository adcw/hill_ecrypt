import random
from string import ascii_uppercase as alphabet
import pandas as pd
import numpy as np
from hill_encrypt import encrypt, decrypt, invert_key
from hill_key import random_key, randomize_key, swap_rows, add_rows_with_random, randomize_rows, smart_rand_rows
from crack_cipher import shotgun_hillclimbing, guess_key_len, guess_key_len
from sklearn.preprocessing import normalize
from time import time
import sys, os


def disablePrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def preprocess_text(text: str, alphabet: str):
    text = text.upper()
    processed = [c for c in text if c in alphabet.upper()]
    return "".join(processed)


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

    text = text * 4
    processed = preprocess_text(text, alphabet)
    letter_data = pd.read_csv("./english_letters.csv")
    freqs = letter_data['frequency'].tolist()

    key = random_key(key_l, alphabet_len)
    print(f"THE KEY: {key}")

    encrypted = encrypt(processed, key, alphabet, freqs)

    cracked_key, old_value = shotgun_hillclimbing(encrypted, key_l, alphabet, freqs=freqs)
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


def determinant_test():
    # Tworzenie macierzy
    r = random_key(3, alphabet_len=len(alphabet))
    A = r.copy()

    # Wyświetlenie wyznacznika oryginalnej macierzy
    print("Wyznacznik macierzy przed modyfikacją: ", np.linalg.det(A))

    # Dodanie drugiego wiersza do pierwszego i trzeciego
    A[0] = A[0] + A[1]
    A[2] = A[2] + A[1]

    # Wyświetlenie wyznacznika macierzy po dodaniu wierszy
    print("Wyznacznik macierzy po dodaniu wierszy: ", np.linalg.det(A))

    A = r.copy()
    # Zamiana drugiego i trzeciego wiersza miejscami
    A[1], A[2] = A[2], A[1]

    # Wyświetlenie wyznacznika macierzy po zamianie wierszy
    print("Wyznacznik macierzy po zamianie wierszy: ", np.linalg.det(A))

    A = r.copy()
    # Zamiana drugiej i trzeciej kolumny miejscami
    A[:, 1], A[:, 2] = A[:, 2], A[:, 1].copy()

    # Wyświetlenie wyznacznika macierzy po zamianie kolumn
    print("Wyznacznik macierzy po zamianie kolumn: ", np.linalg.det(A))

    A = r.copy()
    # Dodanie do pierwszego wiersza wartości drugiego wiersza pomnożonego przez 2
    A[0] = A[0] + 2 * A[1]

    # Wyświetlenie wyznacznika macierzy po dodaniu wiersza pomnożonego przez skalar
    print("Wyznacznik macierzy po dodaniu wiersza pomnożonego przez skalar: ", np.linalg.det(A))


def smart_swap_test():
    key_len = 4
    alphabet_len = len(alphabet)
    key = random_key(key_len, alphabet_len)

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
    encrypted = encrypt(processed, key, alphabet, freqs)

    key_changed = randomize_rows(invert_key(key, alphabet_len), 0.1, 0.5, alphabet_len)
    decrypted = decrypt(encrypted, key, alphabet, freqs)
    decrypted_err = decrypt(encrypted, invert_key(key_changed, alphabet_len), alphabet, freqs)

    with open('./english_bigrams.txt', 'r') as file:
        content = file.readlines()
        splitted = np.array([line.replace("\n", "").split(" ") for line in content])
        splitted[:, 1] = normalize([splitted[:, 1]])
        d = {k: float(v) for k, v in splitted}

    fixed = smart_rand_rows(key_changed, encrypted, alphabet, d, freqs)

    fixed_text = encrypt(encrypted, fixed, alphabet, freqs)

    pass


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

    from utils import mod_inverse_matrix
    itr = 0
    t0 = time()
    while time() - t0 < t_limit:
        mod_inverse_matrix(key, 26)
        itr += 1
    print(f'mod_inverse_matrix: {itr}')

    itr = 0
    disablePrint()
    t0 = time()
    while time() - t0 < t_limit:
        random_key(key_len, alphabet_len)
        itr += 1
    enablePrint()
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

    from numpy import matmul, matrix
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


if __name__ == '__main__':
    # swap_rows_test()
    # crack_test()

    key_l = 3
    alphabet_len = len(alphabet)
    text = 'Attach files by dragging & dropping, selecting or pasting them.'
    processed = preprocess_text(text, alphabet)
    letter_data = pd.read_csv("./english_letters.csv")
    freqs = letter_data['frequency'].tolist()

    key = random_key(key_l, alphabet_len)
    encrypted = encrypt(processed, key, alphabet, freqs)
    inv_key = invert_key(key, alphabet_len)
    decrypted = encrypt(encrypted, inv_key, alphabet, freqs)

    # guess_me_keys_test()
    # crack_test()
    # determinant_test()
    perfomence_test()
    # smart_swap_test()

    # key = random_key(5, 26)
    # changed = add_rows_with_random(key, alphabet_len=26)
    #  Optimizations
    pass
