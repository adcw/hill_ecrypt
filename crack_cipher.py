import random
import threading as thr
from time import time

from math import exp
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm

from hill_encrypt import encrypt, invert_key
from hill_key import random_key, randomize_rows, add_rows_with_random, smart_rand_rows, swap_rows, slide_key
from ngram import Ngram_score as ns
from utils import disable_print, enable_print


def guess_key_len(text: str, alphabet: str, test_time: int = 60 * 3, freqs: list[float] | None = None):
    table = []

    # disable_print()

    def test(i):
        matrix, value = shotgun_hillclimbing(text, i, alphabet, test_time, freqs=freqs, buffer_len=5,
                                             no_progres_bar=True)
        table.append([matrix, value, i])
        pass

    threads = []
    for i in range(2, 11):
        threads.append(
            thr.Thread(target=test, args=[i])
        )
        threads[-1].start()
    for t in threads:
        t.join()
    enable_print()
    table.sort(key=lambda row: (row[1]), reverse=True)
    return table


def shotgun_hillclimbing(text: str, key_len: int, alphabet: str, t_limit: int = 60 * 5, j_max: int = 2000,
                         freqs: list[float] | None = None, start_key: np.matrix | None = None, buffer_len: int = 5,
                         no_progres_bar: bool = False):
    scorer = ns('./english_bigrams.txt')

    with open('./english_bigrams.txt', 'r') as file:
        content = file.readlines()
        splitted = np.array([line.replace("\n", "").split(" ") for line in content])
        splitted[:, 1] = normalize([splitted[:, 1]])
        bigram_data = {k: float(v) for k, v in splitted}

    alphabet_len = len(alphabet)
    if start_key is not None:
        key_old = start_key
    else:
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
            perc = (1 - ((time() - t0) / t_limit)) ** 4

        r = random.random()

        if r < 0.8:
            key_new = randomize_rows(key_old, perc_rows=perc / 2, perc_elems=perc, alphabet_len=alphabet_len)
        elif r < 0.9:
            key_new = swap_rows(key_old)
        else:
            key_new = slide_key(key_old, alphabet_len)


        # r = random.random()
        # if perc < 0.9:
        #     key_new = randomize_rows(key_old, 0.5, 0.5, alphabet_len)
        #     pass
        # else:
        #     if r > 0.3:
        #         key_new = randomize_rows(key_old, 0.5, 0.5, alphabet_len)
        #     elif r > 0.2:
        #         key_new = randomize_rows(key_old, 0.1, 0.1, alphabet_len)
        #     else:
        #         key_new = slide_key(key_old, alphabet_len)
        #     pass
        #
        # if r > 0.6:
        #     key_new = swap_rows(key_new)

        decoded_new = encrypt(text, key_new, alphabet, freqs)
        value_new = scorer.score(decoded_new)

        if value_new > value_old:
            print(f"decoded: {decoded_new[:25]}, value: {value_new / wordlen}, perc = {perc}, key = {key_new}")

            if value_new / wordlen > -2.4:
                print(f'BEST: {decoded_new}, key = \n{key_new}')
                key_old = key_new
                value_old = value_new
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
                percentile = np.percentile(elements, 75)

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

        # print(f"BEST SOLUTION: {encrypt(text, key_old, alphabet, freqs)}")
        # print("IMPROVING...")
        # for _ in tqdm(range(6000), disable=no_progres_bar):
        #     key_new = smart_rand_rows(key_old, text, alphabet, bigram_data, freqs)
        #     decoded_new = encrypt(text, key_new, alphabet, freqs)
        #     value_new = scorer.score(decoded_new)
        #
        #     if value_new > value_old:
        #         print(f"decoded: {decoded_new[:25]}, value: {value_new / wordlen}, perc = {perc}, key = {key_new}")
        #         key_old = key_new.copy()
        #         value_old = value_new
        #
        # best_results[0] = value_old, key_old

        print(f'{buffer_len} best results:')
        for x in best_results[:buffer_len]:
            print(f"{x[0] / wordlen} {x[1]}, {encrypt(text, x[1], alphabet, freqs)[:20]}")

    print(f'elapsed time is {time() - t0} sec, used {itr} iterations')

    real_key = invert_key(key_old, alphabet_len)
    return real_key, value_old


def annealing_multithread():
    pass


def acc_func(diff, t):
    return exp(-diff / t)


# hillclimbing
def annealing(cypher: str, key_len: int, alphabet: str, freqs: list[float] | None = None):
    T = 100  # 100
    dT = -0.005
    alphabet_len = len(alphabet)

    scorer = ns("./english_bigrams.txt")

    key_old = random_key(key_len, alphabet_len)
    decrypted = encrypt(cypher, key_old, alphabet, freqs)
    value_old = scorer.score(decrypted)

    t0 = time()
    t = T
    cypher_len = len(cypher)

    # j = 0

    perc = 1.
    while t > 0:
        if perc <= 0.01:
            perc = 0.01
        else:
            perc = (t / T) ** 4

        for i in range(1):
            r = random.random()

            if r < 0.5:
                key_new = randomize_rows(key_old, perc_rows=perc / 2, perc_elems=perc, alphabet_len=alphabet_len)
            else:
                # elif r < 0.6:
                key_new = swap_rows(key_old)
            # else:
            # key_new = slide_key(key_old, alphabet_len)

            decrypted = encrypt(cypher, key_new, alphabet, freqs)
            value_new = scorer.score(decrypted)

            if value_new > value_old:
                value_old = value_new
                key_old = key_new
            elif random.random() < acc_func(value_old - value_new, t):

                if value_old - value_new > 200:
                    print(f"{perc:.2f}%: {decrypted[:20]}, score: {value_new / cypher_len:.2f}")
                #     print(value_old, ' -> ', value_new)
                j = 0
                value_old = value_new
                key_old = key_new

            if value_old / cypher_len > -2.40:
                print('Seems we found solution in ', time() - t0, 'sec :\nkey =',
                      key_old, decrypted)
                break
        t += dT

    return key_old
