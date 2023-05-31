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


def crack_test():
    key_l = 5
    alphabet_len = len(alphabet)

    with open("./text.txt", "r") as file:
        text = file.read()

    processed = preprocess_text(text, alphabet)
    letter_data = read_csv("./english_letters.csv")
    freqs = letter_data['frequency'].tolist()
    key = random_key(key_l, alphabet_len)
    print(f"The key: \n{key}\n, ITS INVERSE: \n{invert_key(key, alphabet_len)}\n")

    encrypted = encrypt(processed, key, alphabet, freqs)

    ngram_file_path = 'english_trigrams.txt'
    # ngram_file_path = 'english_bigrams.txt'

    """
    Best bend values
    key_len | row bend | elem bend | times in s
    2       | 1.40     | 0.99      | 0.13, 8.82, 10.89, 24.35(ała), 83.71(bardzo wredny klucz)
    3       | 1.9      | 0.9       | 99.42, 162.98 237.22
    4       | 2        | 1.1       | 720.50
    5       | 4        | 1.5       | did not resolve in 2 hours
    2       | 1.3      | 0.8       |
    
    
    5, trigram: 0.11940322755261186 perc
    """

    cracked_text, cracked_key = crack_cipher.crack(cypher=encrypted, alphabet=alphabet,
                                                   bigram_file_path='english_bigrams.txt',
                                                   ngram_file_path=ngram_file_path,
                                                   freqs=freqs)

    print(f"THIS IS CRACKED TEXT: {cracked_text}")

    pass


def guess_me_keys_test():
    """Guessing between 3 and 5"""
    key_l = 3
    alphabet_len = len(alphabet)

    with open("./text.txt", "r") as file:
        text = file.read()

    processed = preprocess_text(text, alphabet)
    letter_data = read_csv("./english_letters.csv")
    freqs = letter_data['frequency'].tolist()
    key = random_key(key_l, alphabet_len)
    ngram_file_path = 'english_trigrams.txt'
    print(f"The key: \n{key}\n, ITS INVERSE: \n{invert_key(key, alphabet_len)}\n")
    encrypted = encrypt(processed, key, alphabet, freqs)
    table = guess_key_len(encrypted, alphabet, freqs=freqs, bigram_file_path='english_bigrams.txt',
                          ngram_file_path=ngram_file_path, t_limit=60 * 5)
    print(table)
    print(f'I guess key length is= {table[0][0].shape[0]}')
    print(f'True key length = {key_l}')

    pass


def test_german():
    german_alphabet, german_bigrams, german_trigrams, german_freqs = utils.get_german()

    with open("german_text_to_crack.txt") as file:
        text = file.read()

    text = preprocess_text(text, german_alphabet)

    key_len = 5
    alphabet_len = len(german_alphabet)
    key = hill_key.random_key(key_len, alphabet_len)

    print(f"KEY:\n{key}, inverse:\n{hill_encrypt.invert_key(key, alphabet_len)}")

    encrypted = encrypt(text, key, german_alphabet, german_freqs)

    cracked_text, cracked_key = crack_cipher.crack(cypher=encrypted, alphabet=german_alphabet,
                                                   bigram_file_path=german_bigrams,
                                                   ngram_file_path=german_trigrams,
                                                   freqs=german_freqs, target_score=-3.5, bad_score=-7,
                                                   print_threshold=-100)

    print(f"THIS IS CRACKED TEXT: {cracked_text}")


# documenting

if __name__ == '__main__':
    # swap_rows_test()
    # determinant_test()
    # perfomence_test()
    # smart_swap_test()
    # test_chunkify_text()
    # change_key_performance()
    # crack_test()
    # guess_me_keys_test()
    # gpu_test()
    # test_inversion()

    # crack_test()
    # test_trigram()
    # test_shotgun(alphabet, n_tests=50)

    # test_ngram_numbers()

    test_german()
    # with open('./german_text_in', encoding="UTF-8") as file:
    #     txt = file.read()
    #     scorer = ngram.Ngram_score("./german_trigrams.txt")
    #     target, bad = crack_cipher.get_scores(txt, "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜß", scorer)
    #
    #     print(f"target = {target}, bad = {bad}")

    # utils.generate_grams('./german_text_in', './german_trigrams.txt', n=3, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜß")
    # utils.generate_grams('./german_text_in', './german_bigrams.txt.txt', n=2, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜß")