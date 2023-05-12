import random
import threading as thr
from multiprocessing import Pool
from mpire import WorkerPool
from time import time

from math import exp
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm

import ngram
from hill_encrypt import encrypt, invert_key, fast_encrypt
from hill_key import random_key, randomize_rows, add_rows_with_random, smart_rand_rows, swap_rows, slide_key, \
    small_change
from ngram import Ngram_score as ns
from utils import disable_print, enable_print
from math import ceil
import winsound


def guess_key_len(text: str, alphabet: str, test_time: int = 60 * 3, freqs: list[float] | None = None):
    table = []

    # disable_print()

    def test(i):
        matrix, value = shotgun_hillclimbing(text, i, alphabet, test_time, freqs=freqs)
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


def upgrade_key_unwraper(i, c):
    value = upgrade_key(c, *i)
    return value


"""
   key_old, value_old, found = upgrade_key(key=key_old, cypher=text, alphabet=alphabet, scorer=scorer, a=a,
                                                        iters=search_deepness,
                                                        row_bend=row_bend,
                                                        elem_bend=elem_bend,
                                                        freqs=freqs,
                                                        bad_score=bad_score,
                                                        target_score=target_score
                                                        )
"""


def upgrade_key(
        key: np.matrix,
        cypher: str,
        alphabet: str,
        scorer: ngram.Ngram_score,
        a: float,
        iters: 100,
        row_bend: float,
        elem_bend: float,

        freqs: list[float] | None,
        bad_score: float = -4,
        target_score: float = -2.4,

):
    alphabet_len = len(alphabet)
    word_len = len(cypher)

    key_old = key.copy()
    found = False
    value_old = scorer.score(encrypt(cypher, key_old, alphabet, freqs))

    for i in range(iters):

        perc: float = (a * (value_old / word_len - bad_score) + 1)
        perc = max(min(perc, 1), 0.01)
        perc_rows = perc ** row_bend
        perc_elems = perc ** elem_bend

        r = random.random()

        if r < 0.8:
            key_new = randomize_rows(key_old, perc_rows=perc_rows, perc_elems=perc_elems,
                                     alphabet_len=alphabet_len)
        elif r < 0.9:
            key_new = swap_rows(key_old)
        else:
            key_new = slide_key(key_old, alphabet_len)

        if random.random() < 0.05:
            key_new = slide_key(key_old, alphabet_len, horizontal=True)

        decoded_new = encrypt(cypher, key_new, alphabet, freqs)
        value_new = scorer.score(decoded_new)

        if value_new > value_old:
            print(
                f"i = {i}, decoded: {decoded_new[:25]}, value: {value_new / word_len}, "
                f"perc_rows = {perc_rows}, perc_elems = {perc_elems} key = {key_new}")

            if value_new / word_len > target_score:
                print(f'BEST: {decoded_new}, key = \n{key_new}')
                found = True
                key_old = key_new
                value_old = value_new
                break

            key_old = key_new
            value_old = value_new

    return key_old, value_old, found


class Notifier:
    def __init__(self, thresholds: list[float], duration: int = 200, freq: int = 800):
        self.t_dict = {k: False for k in thresholds}
        self.duration = duration
        self.freq = freq

    def update(self, score):
        threshold_passed = None
        for threshold in self.t_dict.keys():
            if score > threshold:
                threshold_passed = threshold
                break

        if not self.t_dict.get(threshold_passed, True):
            for _ in range(10):
                winsound.Beep(self.freq, self.duration)
            self.t_dict[threshold_passed] = True


def shotgun_hillclimbing(text: str,
                         key_len: int,
                         alphabet: str,

                         t_limit: int = 60 * 5,
                         search_deepness: int = 1000,
                         freqs: list[float] | None = None,
                         start_key: np.matrix | None = None,
                         target_score: int = -2.4,
                         row_bend: float = 1,
                         elem_bend: float = 1,
                         sound: bool = False
                         ):
    scorer = ns('./english_bigrams.txt')

    alphabet_len = len(alphabet)
    if start_key is not None:
        key_old = start_key
    else:
        key_old = random_key(key_len=key_len, alphabet_len=alphabet_len)
    value_old = scorer.score(encrypt(text, key_old, alphabet, freqs))

    t0, itr, j = time(), 0, 0
    word_len = len(text)

    bad_score = -3.6
    a = -0.99 / (target_score - bad_score)
    it_args = [key_old] * (key_len * 10)
    args = [text, alphabet, scorer, a, search_deepness, row_bend, elem_bend, freqs, bad_score, target_score]

    # Create notifier, give a list of thresholds after which the beeping occurs.
    notifier = Notifier([-3])

    with WorkerPool(n_jobs=10, shared_objects=args, keep_alive=True, daemon=True) as pool:
        while time() - t0 < t_limit:
            if key_len > 2:
                table = pool.map(upgrade_key_unwraper, iterable_of_args=it_args)

                table.sort(key=lambda row: (row[1]), reverse=True)

                score = table[0][1] / word_len

                # Update the notifier
                notifier.update(score)

                if table[0][2]:
                    t = time() - t0
                    print(f"time: {t:.2f}, iters: {itr}, {itr / t:.2f}it/s")
                    return invert_key(table[0][0], alphabet_len), table[0][1]

                it_args = [row[0] for row in table[:int(ceil(len(table) / 4))]] * 4
                print("ITERACJA Wątkowa")
            else:
                key_old, value_old, found = upgrade_key(key=key_old, cypher=text, alphabet=alphabet, scorer=scorer, a=a,
                                                        iters=search_deepness,
                                                        row_bend=row_bend,
                                                        elem_bend=elem_bend,
                                                        freqs=freqs,
                                                        bad_score=bad_score,
                                                        target_score=target_score
                                                        )
                if found:
                    t = time() - t0
                    print(f"time: {t:.2f}, iters: {itr}, {itr / t:.2f}it/s")
                    return invert_key(key_old, alphabet_len), value_old

        t = time() - t0
        print(f"time: {t:.2f}, iters: {itr}, {itr / t:.2f}it/s")

        if key_len > 2:
            return invert_key(table[0][0], alphabet_len), table[0][1]
        else:
            real_key = invert_key(key_old, alphabet_len)
            return real_key, value_old


def fast_shotgun(chunkified_text: list[list[int]],
                 key_len: int,
                 alphabet_len: int,
                 scorer: ngram.NgramNumbers,
                 t_limit: int = 60 * 5,
                 j_max: int = 2000,
                 start_key: np.matrix | None = None,
                 buffer_len: int = 5,
                 target_fitness: float = -2.4):
    key_old = random_key(key_len, alphabet_len) if start_key is None else start_key
    value_old = scorer.chunklist_score(fast_encrypt(chunkified_text, key_old, alphabet_len))

    best_results = []
    t0, itr, j = time(), 0, 0
    chunkified_text_shape = np.shape(chunkified_text)
    word_len = chunkified_text_shape[0] * chunkified_text_shape[1]
    perc = 1

    bad_fitness = -3.6
    a = -0.99 / (target_fitness - bad_fitness)

    while time() - t0 < t_limit:
        # if perc <= 0.01:
        #     perc = 0.01
        # else:
        #     perc = (1 - ((time() - t0) / t_limit)) ** 10
        perc = np.clip((a * (value_old / word_len - bad_fitness) + 1) ** 0.3, 0.01, 1)  # if key_len != 2 else 1

        r = random.random()

        if r < 0.7:
            key_new = randomize_rows(key_old, perc_rows=perc * 1, perc_elems=perc * 1, alphabet_len=alphabet_len)
            # key_new = small_change(key_old, alphabet_len)
        elif r < 0.8:
            key_new = swap_rows(key_old)
        else:
            key_new = slide_key(key_old, alphabet_len)

        if random.random() < 0.1:
            key_new = slide_key(key_old, alphabet_len, horizontal=True)

        value_new = scorer.chunklist_score(fast_encrypt(chunkified_text, key_new, alphabet_len))

        if value_new > value_old:
            print(f"perc: {perc}, value: {value_new / word_len}, key:")
            print(key_new)

            if value_new / word_len > target_fitness:
                print("SOLUTION FOUND")
                print(f'elapsed time is {time() - t0} sec, used {itr} iterations')
                return invert_key(key_new, alphabet_len), value_new

            # key_old, value_old = key_new.copy(), value_new
            key_old = key_new
            value_old = value_new
            j = 0

        if j > j_max:

            if len(best_results) <= buffer_len:
                best_results.append((value_old, key_old))

                key_old = random_key(key_len, alphabet_len)
                value_old = scorer.chunklist_score(fast_encrypt(chunkified_text, key_old, alphabet_len))
            else:
                # elements = [x[0] for x in best_results]
                # percentile = np.percentile(elements, 75)

                # if value_old > percentile:
                best_results.append((value_old, key_old))
                best_results.sort(reverse=True, key=lambda t: t[0])
                best_results = best_results[:buffer_len]

                if random.random() < 0.95:
                    value_old, key_old = random.choice(best_results[:buffer_len])
                else:
                    key_old = random_key(key_len, alphabet_len)
                    value_old = scorer.chunklist_score(fast_encrypt(chunkified_text, key_old, alphabet_len))

        itr += 1
        j += 1

    if len(best_results) > 0:
        best_results.sort(reverse=True, key=lambda x_: x_[0])
        value_old, key_old = best_results[0]

    t = time() - t0
    print(f'elapsed time is {t:.2f} sec, used {itr} iterations. Speed: {itr / t:.2f} its/s')

    real_key = invert_key(key_old, alphabet_len)
    return real_key, value_old

    # perc: float = (a * (value_old / word_len - bad_score) + 1)
    #     # perc = max(min(perc, 1), 0.01)
    #     #
    #     # r = random.random()
    #     #
    #     # if r < 0.8:
    #     #     key_new = randomize_rows(key_old, perc_rows=perc ** 1.2, perc_elems=perc ** 0.6,
    #     #                              alphabet_len=alphabet_len)
    #     # elif r < 0.9:
    #     #     key_new = swap_rows(key_old)
    #     # else:
    #     #     key_new = slide_key(key_old, alphabet_len)
    #     #
    #     # if random.random() < 0.05:
    #     #     key_new = slide_key(key_old, alphabet_len, horizontal=True)
    #     #
    #     # decoded_new = encrypt(text, key_new, alphabet, freqs)
    #     # value_new = scorer.score(decoded_new)
    #     # #
    #     # if value_new > value_old:
    #     #     print(
    #     #         f"i = {i}, decoded: {decoded_new[:25]}, value: {value_new / word_len}, perc = {perc}, key = {key_new}")
    #     #
    #     #     if value_new / word_len > target_score:
    #     #         print(f'BEST: {decoded_new}, key = \n{key_new}')
    #     #         key_old = key_new
    #     #         value_old = value_new
    #     #         found = True
    #     #         break
    #     #
    #     #     key_old = key_new
    #     #     value_old = value_new
    #
    #     if found:
    #         break
    #
    #     # if j > j_max:
    #     if len(best_results) <= buffer_len:
    #         best_results.append((value_old, key_old))
    #
    #         key_old = random_key(key_len=key_len, alphabet_len=alphabet_len)
    #         value_old = scorer.score(encrypt(text, key_old, alphabet, freqs))
    #     else:
    #         min_indx = 0
    #         min_val = best_results[0][0]
    #         for i in range(1, buffer_len):
    #             v, _ = best_results[i]
    #             if v < min_val:
    #                 min_indx = i
    #                 min_val = v
    #
    #         best_results[min_indx] = (value_old, key_old)
    #
    #         if random.random() < 0.95:
    #             value_old, key_old = random.choice(best_results)
    #             # key_old = key_old.copy()
    #         else:
    #             key_old = random_key(key_len=key_len, alphabet_len=alphabet_len)
    #             value_old = scorer.score(encrypt(text, key_old, alphabet, freqs))
    #
    #     itr += 1
    #     # j += 1
    #
    # if not found and len(best_results) > 0:
    #     best_results.sort(reverse=True, key=lambda x_: x_[0])
    #     value_old, key_old = best_results[0]
    #
    #     print(f'{buffer_len} best results:')
    #     for x in best_results[:buffer_len]:
    #         print(f"{x[0] / word_len} {x[1]}, {encrypt(text, x[1], alphabet, freqs)[:20]}")
    #
    # t = time() - t0
    # print(f'elapsed time is {t:.2f} sec, used {itr} iterations. Speed: {itr / t:.2f} its/s, q = {value_old}')
