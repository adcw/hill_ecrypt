import numpy as np

from ngram import Ngram_score as ns
from hill_key import random_key, randomize_key, randomize_rows, add_rows_with_random, smart_rand_rows
from hill_encrypt import encrypt, invert_key
from time import time
import random
from sklearn.preprocessing import normalize


def shotgun_hillclimbing(text: str, key_len: int, alphabet: str, t_limit: int = 60 * 20, j_max: int = 2000,
                         freqs: list[float] | None = None, buffer_len: int = 10):
    scorer = ns('./english_bigrams.txt')

    with open('./english_bigrams.txt', 'r') as file:
        content = file.readlines()
        splitted = np.array([line.replace("\n", "").split(" ") for line in content])
        splitted[:, 1] = normalize([splitted[:, 1]])
        bigram_data = {k: float(v) for k, v in splitted}

    alphabet_len = len(alphabet)
    key_old = random_key(key_len=key_len, alphabet_len=alphabet_len)
    value_old = scorer.score(encrypt(text, key_old, alphabet, freqs))

    best_results = []
    t0, itr, j = time(), 0, 0

    wordlen = len(text)

    perc = 1
    found = False

    while time() - t0 < t_limit:
        if perc <= 0.01:
            perc = 0.01
        else:
            perc = (1 - ((time() - t0) / t_limit)) ** 0.9

        if 1 >= perc > 0.6:
            if random.random() < 0.8:
                key_new = add_rows_with_random(key_old, alphabet_len=alphabet_len)
            else:
                key_new = randomize_rows(key_old, perc_rows=perc / 2, perc_elems=perc, alphabet_len=alphabet_len)
        elif 0.6 >= perc > 0.4:
            if random.random() < 0.8:
                key_new = smart_rand_rows(key_old, text, alphabet, bigram_data, freqs)
            else:
                key_new = randomize_rows(key_old, perc_rows=perc / 2, perc_elems=perc, alphabet_len=alphabet_len)
        else:
            key_new = smart_rand_rows(key_old, text, alphabet, bigram_data, freqs)

        decoded_new = encrypt(text, key_new, alphabet, freqs)
        value_new = scorer.score(decoded_new)

        if value_new > value_old:
            print(f"decoded: {decoded_new[:25]}, value: {value_new / wordlen}, perc = {perc}, key = {key_new}")

            if value_new / wordlen > -2.4:
                print(f'BEST: {decoded_new}, key = \n{key_new}')
                key_old = key_new
                found = True
                break
            key_old, value_old = key_new.copy(), value_new
            j = 0

        if j > j_max:

            if len(best_results) <= buffer_len:
                best_results.append((value_old, key_old))

                key_old = random_key(key_len=key_len, alphabet_len=alphabet_len)
                value_old = scorer.score(encrypt(text, key_old, alphabet, freqs))
            else:
                elements = [x[0] for x in best_results]
                percentile = np.percentile(elements, 0.75)

                if value_old > percentile:
                    best_results.append((value_old, key_old))
                    best_results.sort(reverse=True, key=lambda t: t[0])
                    best_results = best_results[:buffer_len]

                if random.random() < 0.95:
                    value_old, key_old = random.choice(best_results[:buffer_len])
                    key_old = key_old.copy()
                else:
                    key_old = random_key(key_len=key_len, alphabet_len=alphabet_len)
                    value_old = scorer.score(encrypt(text, key_old, alphabet, freqs))

        itr += 1
        j += 1

    if not found and len(best_results) > 0:
        best_results.sort(reverse=True, key=lambda x_: x_[0])
        value_old, key_old = best_results[0]

        print(f'{buffer_len} best results:')
        for x in best_results[:buffer_len]:
            print(f"{x[0] / wordlen} {x[1]}, {encrypt(text, x[1], alphabet, freqs)[:20]}")

    print(f'elapsed time is {time() - t0} sec, used {itr} iterations')

    real_key = invert_key(key_old, alphabet_len)
    return real_key
