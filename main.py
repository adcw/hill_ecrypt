import numpy as np
from numpy import matrix, array
from string import ascii_uppercase as alphabet
import pandas as pd
import numpy as np
from hill_encrypt import encrypt, decrypt
from hill_key import random_key, randomize_key, swap_rows, invert_key
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
    text = 'This book is about testing, experimenting, ' \
           'and playing with language.It is a handbook of ' \
           'tools and techniques for taking words apart and putting them back together again in ways that ' \
           'I hope are meaningful and legitimate ( or even illegitimate).This book is about peeling back ' \
           'layers in search of the language-making energy of the human spirit.It is about the gaps in ' \
           'meaning that we urgently need to notice and name—the places where our dreams and ideals are ' \
           'no longer fulfilled by a society that has become fast-paced and hyper-commercialized.'
    letter_data = pd.read_csv("./english_letters.csv")
    freqs = letter_data['frequency'].tolist()

    key = random_key(key_l, alphabet_len)
    encrypted = encrypt(text, key, alphabet, freqs)

    cracked_key = shotgun_hillclimbing(encrypted, key_l, alphabet, freqs=freqs, buffer_len=7)
    cracked_text = decrypt(encrypted, cracked_key, alphabet, freqs)

    pass

def determinant_test():
    # Tworzenie macierzy
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Wyświetlenie wyznacznika oryginalnej macierzy
    print("Wyznacznik macierzy przed modyfikacją: ", np.linalg.det(A))

    # Dodanie drugiego wiersza do pierwszego i trzeciego
    A[0] = A[0] + A[1]
    A[2] = A[2] + A[1]

    # Wyświetlenie wyznacznika macierzy po dodaniu wierszy
    print("Wyznacznik macierzy po dodaniu wierszy: ", np.linalg.det(A))

    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # Zamiana drugiego i trzeciego wiersza miejscami
    A[1], A[2] = A[2], A[1]

    # Wyświetlenie wyznacznika macierzy po zamianie wierszy
    print("Wyznacznik macierzy po zamianie wierszy: ", np.linalg.det(A))


    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # Zamiana drugiej i trzeciej kolumny miejscami
    A[:, 1], A[:, 2] = A[:, 2], A[:, 1].copy()

    # Wyświetlenie wyznacznika macierzy po zamianie kolumn
    print("Wyznacznik macierzy po zamianie kolumn: ", np.linalg.det(A))

    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # Dodanie do pierwszego wiersza wartości drugiego wiersza pomnożonego przez 2
    A[0] = A[0] + 2 * A[1]

    # Wyświetlenie wyznacznika macierzy po dodaniu wiersza pomnożonego przez skalar
    print("Wyznacznik macierzy po dodaniu wiersza pomnożonego przez skalar: ", np.linalg.det(A))

    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    A[2] += 2 * A[0]
    print("Wyznacznik macierzy po dodanie do jednego wiersza innego wiersza przemnożonego przez stałą: ", np.linalg.det(A))

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

    # crack_test()
    determinant_test()

    pass
