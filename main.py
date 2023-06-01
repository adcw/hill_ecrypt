from string import ascii_uppercase as alphabet

from pandas import read_csv

import crack_cipher
import hill_encrypt
import hill_key
import utils
from crack_cipher import guess_key_len
from hill_encrypt import encrypt
from hill_encrypt import invert_key
from hill_key import random_key
from utils import preprocess_text

"""
Szyfr Hilla
Przemysław Kawa, Adrian Ćwiąkała

Do łamania szyfru wykorzystano klasyczną metodę shotgun z kilkoma modyfikacjami
- metoda wspinaczki generuje odwrotność klucza, więc wynikowy klucz jest jego odwrotnością.

- wspinaczka realozowana jest na wielu procesach.

- procent zmiany klucza maleje wraz ze wzrostem jakości odszyfrowywanego nim tekstu, dzięki czemu na samym początku
  klucz jest zmieniany w stopniu znacznym, a pod koniec tylko nieznacznie.
  
- Jeśli zbliżamy się do rozwiązania, uruchamiana jest funkcja oceniająca, które wiersze należy zmienić by złamać klucz.
  Metoda ta bazuje na obliczaniu jakości metodą bigram i ocenianiu które litery odszyfrowanego tekstu są już poprawne,
  a które należy dalej zmieniać. Zmieniane są tylko te wiersze, które generują litery nie pasujące do pozostałych.
"""


def crack_test():
    key_l = 5
    alphabet_len = len(alphabet)

    with open("./text.txt", "r", encoding="UTF-8") as file:
        text = file.read()

    processed = preprocess_text(text, alphabet)
    letter_data = read_csv("language_data/english_letters.csv")
    freqs = letter_data['frequency'].tolist()
    key = random_key(key_l, alphabet_len)
    print(f"The key: \n{key}\n, ITS INVERSE: \n{invert_key(key, alphabet_len)}\n")

    encrypted = encrypt(processed, key, alphabet, freqs)

    ngram_file_path = 'language_data/english_trigrams.txt'
    # ngram_file_path = 'english_bigrams.txt'

    """
    Best bend values
    key_len | row bend | elem bend | times in s
    2       | 1.40     | 0.99      | 0.13, 8.82, 10.89, 24.35(ała), 83.71(bardzo wredny klucz)
    3       | 1.9      | 0.9       | 99.42, 162.98 237.22
    4       | 2        | 1.1       | 720.50
    5       | 4        | 1.5       | 
    2       | 1.3      | 0.8       |
    
    """

    cracked_text, cracked_key = crack_cipher.crack(cypher=encrypted, alphabet=alphabet,
                                                   bigram_file_path='language_data/english_bigrams.txt',
                                                   ngram_file_path=ngram_file_path,
                                                   freqs_file_path=freqs)

    print(f"THIS IS CRACKED TEXT: {cracked_text}")

    pass


def guess_me_keys_test():
    """Guessing between 3 and 5"""
    key_l = 3
    alphabet_len = len(alphabet)

    with open("./text.txt", "r") as file:
        text = file.read()

    processed = preprocess_text(text, alphabet)
    letter_data = read_csv("language_data/english_letters.csv")
    freqs = letter_data['frequency'].tolist()
    key = random_key(key_l, alphabet_len)
    ngram_file_path = 'language_data/english_trigrams.txt'
    print(f"The key: \n{key}\n, ITS INVERSE: \n{invert_key(key, alphabet_len)}\n")
    encrypted = encrypt(processed, key, alphabet, freqs)
    table = guess_key_len(encrypted, alphabet, freqs=freqs, bigram_file_path='language_data/english_bigrams.txt',
                          ngram_file_path=ngram_file_path, t_limit=60 * 5)
    print(table)
    print(f'I guess key length is= {table[0][0].shape[0]}')
    print(f'True key length = {key_l}')

    pass


def test_crack(alph: str = alphabet,
               bigram_file_path: str = './language_data/english_bigrams.txt',
               ngram_file_path: str = './language_data/english_trigrams.txt',
               letter_freqs_file_path: str = './language_data/english_letter_freqs.txt',
               text_to_crack_path: str = './english_text_to_crack.txt'):
    with open(text_to_crack_path, encoding="UTF-8") as file:
        text = file.read()

    freqs = utils.parse_freqs(freqs_file_path=letter_freqs_file_path)

    text = preprocess_text(text, alphabet=alph)

    key_len = 3
    alphabet_len = len(alph)
    key = hill_key.random_key(key_len, alphabet_len)

    print(f"KEY:\n{key}, inverse:\n{hill_encrypt.invert_key(key, alphabet_len)}")

    encrypted = encrypt(text, key, alph, freqs)

    cracked_text, cracked_key = crack_cipher.crack(cypher=encrypted, alphabet=alph,
                                                   bigram_file_path=bigram_file_path,
                                                   ngram_file_path=ngram_file_path,
                                                   freqs_file_path=freqs, target_score=-3.5, bad_score=-7,
                                                   print_threshold=-100)

    with open("./output.txt", mode="w+", encoding="UTF-8") as file:
        file.write(cracked_text)
    print(f"Cracked text was saved to the file.")


if __name__ == '__main__':
    # Łamanie tekstu angielskiego
    # test_crack()

    # Łamanie tekstu francuskiego
    french_alphabet = utils.get_alphabet('language_data/french_alphabet.txt')

    test_crack(alph=french_alphabet,
               bigram_file_path="language_data/french_bigrams.txt",
               ngram_file_path="language_data/french_trigrams.txt",
               letter_freqs_file_path="language_data/french_letter_freqs.txt",
               text_to_crack_path="french_text_to_crack.txt")

    # utils.generate_grams("./language_data/french_text_sample.txt", "./language_data/french_bigrams.txt",
    #                      "./language_data/french_alphabet.txt", 2)
    #
    # utils.generate_grams("./language_data/french_text_sample.txt", "./language_data/french_trigrams.txt",
    #                      "./language_data/french_alphabet.txt", 3)
    #
    # utils.genereate_freqs("./language_data/french_text_sample.txt", "./language_data/french_letter_freqs.txt",
    #                       french_alphabet)
