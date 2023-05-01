import numpy as np

from ngram import Ngram_score as ns
from hill_key import random_key, invert_key, randomize_key, randomize_rows, add_rows_with_random
from hill_encrypt import encrypt
from time import time
import random
from math import ceil


def shotgun_hillclimbing(text: str, key_len: int, alphabet: str, t_limit: int = 60 * 10, j_max: int = 6000,
                         freqs: list[float] | None = None, buffer_len: int = 10):
    scorer = ns('./english_bigrams.txt')

    alphabet_len = len(alphabet)
    key_old = random_key(key_len=key_len, alphabet_len=alphabet_len)
    value_old = scorer.score(encrypt(text, key_old, alphabet, freqs))

    best_results = []
    t0, itr, j = time(), 0, 0

    wordlen = len(text)

    perc = 1

    while time() - t0 < t_limit:
        if perc <= 0.01:
            perc = 0.01
        else:
            perc = (1 - ((time() - t0) / t_limit))
        if random.random() < 0.10:
            key_new = add_rows_with_random(key_old, alphabet_len=alphabet_len)
        else:
            key_new = randomize_rows(key_old, perc_rows=perc / 2, perc_elems=perc, alphabet_len=alphabet_len)
        decoded_new = encrypt(text, key_new, alphabet, freqs)
        value_new = scorer.score(decoded_new)

        if value_new > value_old:
            print(f"decoded: {decoded_new[:25]}, value: {value_new / wordlen}, perc = {perc}")

            if value_new / wordlen > -2.4:
                print(f'BEST: {decoded_new}')
                break
            key_old, value_old = key_new, value_new
            j = 0

        if j > j_max:
            best_results.append((value_old, key_old))
            if len(best_results) >= buffer_len:
                best_results.sort(reverse=True, key=lambda t: t[0])

                if best_results[0][0] > -3:
                    break

                # if best_results[0][1] == best_results[1][1] == best_results[2][1]:
                #     break
                first_matrix = best_results[0][1]
                all_same = True
                for entry in best_results[1:]:
                    mat = entry[1]
                    if not np.array_equal(first_matrix, mat):
                        all_same = False
                        break

                if all_same:
                    break

                if random.random() < 0.95:
                    value_old, key_old = random.choice(best_results[:int(ceil(buffer_len * (perc ** 0.15)))])
                else:
                    key_old = random_key(key_len=key_len, alphabet_len=alphabet_len)
                    value_old = scorer.score(encrypt(text, key_old, alphabet, freqs))
            else:
                key_old = random_key(key_len=key_len, alphabet_len=alphabet_len)
                value_old = scorer.score(encrypt(text, key_old, alphabet, freqs))
        itr += 1
        j += 1
    if len(best_results) > 0:
        best_results.sort(reverse=True, key=lambda x_: x_[0])
        value_old, key_old = best_results[0]

        print(f'{buffer_len} best results:')
        for x in best_results[:buffer_len]:
            print(x)

    print(f'elapsed time is {time() - t0} sec, used {itr} iterations')
    # print(decode_dict(text, key_old), value_old)
    print(encrypt(text, key_old, alphabet, freqs), value_old)

    return invert_key(key_old, alphabet_len)
