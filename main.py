from string import ascii_uppercase as alphabet

import pandas as pd

from crack_cipher import shotgun_hillclimbing, guess_key_len
from hill_encrypt import encrypt, decrypt
from hill_encrypt import invert_key
from hill_key import random_key
from utils import preprocess_text


def crack_test():
    key_l = 4
    alphabet_len = len(alphabet)

    with open("./text.txt", "r") as file:
        text = file.read()

    processed = preprocess_text(text, alphabet)
    letter_data = pd.read_csv("./english_letters.csv")
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

    cracked_key, a = shotgun_hillclimbing(encrypted, key_l, alphabet,
                                          ngram_file_path=ngram_file_path,
                                          freqs=freqs,
                                          bigram_file_path='english_bigrams.txt',
                                          t_limit=60 * 120,
                                          target_score=-3.7,
                                          bad_score=-5.8,
                                          print_threshold=-5.5,
                                          search_deepness=1000,
                                          row_bend=1.5,
                                          elem_bend=1.2,
                                          sound_thresholds=[5, 4.5, 4],
                                          sound=False)

    cracked_text = decrypt(encrypted, cracked_key, alphabet, freqs)
    print(f"Cracked text: {cracked_text}")

    pass


def guess_me_keys_test():
    """Guessing between 3 and 6"""
    key_l = 4
    alphabet_len = len(alphabet)

    with open("./text.txt", "r") as file:
        text = file.read()

    processed = preprocess_text(text, alphabet)
    letter_data = pd.read_csv("./english_letters.csv")
    freqs = letter_data['frequency'].tolist()
    key = random_key(key_l, alphabet_len)
    ngram_file_path = 'english_trigrams.txt'
    print(f"The key: \n{key}\n, ITS INVERSE: \n{invert_key(key, alphabet_len)}\n")
    encrypted = encrypt(processed, key, alphabet, freqs)
    table = guess_key_len(encrypted, alphabet, freqs=freqs, bigram_file_path='english_bigrams.txt',
                          ngram_file_path=ngram_file_path, )
    print(table)
    print(f'I guess key length is= {table[0][0].shape[0]}')
    print(f'True key length = {key_l}')

    pass


# documenting

if __name__ == '__main__':
    # swap_rows_test()
    # determinant_test()
    # perfomence_test()
    # smart_swap_test()
    # test_chunkify_text()
    # change_key_performance()
    crack_test()
    # guess_me_keys_test()
    # gpu_test()
    # test_inversion()
    # test_smart_rand()
    # test_trigram()
    # test_shotgun(alphabet, n_tests=50)

    # test_ngram_numbers()

    pass
