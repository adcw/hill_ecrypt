import numpy as np
from numpy import matrix, array
from string import ascii_uppercase as alphabet
import pandas as pd
import numpy as np
from hill_encrypt import encrypt, decrypt
from hill_key import random_key, randomize_key, swap_rows, add_rows_with_random, invert_key
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
           'building their nests on my boughs, and when the wind blew, I should bow with' \
           '    stately dignity like my tall companions." The tree was so discontented, that it" \
            "took no pleasure in the warm sunshine, the birds, or the rosy clouds that floated' \
           'over it morning and evening. Sometimes, in winter, when the snow lay white and' \
           'glittering on the ground, a hare would come springing along, and jump right over' \
           'the little tree; and then how mortified it would feel!'

    letter_data = pd.read_csv("./english_letters.csv")
    freqs = letter_data['frequency'].tolist()

    key = random_key(key_l, alphabet_len)
    print(f"THE KEY: {key}")

    encrypted = encrypt(text, key, alphabet, freqs)

    cracked_key = shotgun_hillclimbing(encrypted, key_l, alphabet, freqs=freqs)
    cracked_text = decrypt(encrypted, cracked_key, alphabet, freqs)

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
    key_len = 6
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

    encrypted = encrypt(text, key, alphabet, freqs)


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
    # determinant_test()

    # key = random_key(5, 26)
    # changed = add_rows_with_random(key, alphabet_len=26)

    pass
