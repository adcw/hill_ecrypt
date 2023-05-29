import random
from math import ceil
from operator import itemgetter
from time import time

import numpy as np
import winsound
from mpire import WorkerPool
from sklearn.preprocessing import normalize
from tqdm import tqdm

import hill_encrypt
import ngram
import utils
from hill_encrypt import encrypt, invert_key
from hill_key import random_key, randomize_rows, smart_rand_rows, swap_rows, slide_key
from ngram import Ngram_score as ns
from utils import disable_print, enable_print


def guess_len_unwrapper(args, key_len, row_bend, elem_bend):
    """
    The unwrapperW
    :param args:
    :param key_len:
    :param row_bend:
    :param elem_bend:
    :return:
    """
    table = single_process_shotgun(key_len, row_bend, elem_bend, *args)
    return table


def crack(cypher: str,
          alphabet: str,
          bigram_file_path: str = 'english_bigrams.txt',
          ngram_file_path: str = 'english_trigrams.txt',
          search_deepness: int = 1000,
          freqs: list[float] | None = None,
          target_score: float = -3.7,
          bad_score: float = -5.8,
          print_threshold: float = -5,
          ) -> tuple[str, np.matrix]:
    """
    Crack the encrypted text.
    :param cypher: Text to decode (to break)
    :param alphabet: The alphabet of the text to crack
    :param bigram_file_path: the file path of bigram file
    :param ngram_file_path: the file path of ngram file
    :param search_deepness: the number of iterations
    :param freqs: the frequencies of letter occurrence
    :param target_score: the target score for cypher to get
    :param bad_score: the score of random text
    :param print_threshold: the minimal score required to print solution to console
    :return:
    """

    print("Trying to crack the text using keys of shape 2x2...")
    disable_print()
    key, value = shotgun_hillclimbing(text=cypher, key_len=2, alphabet=alphabet, ngram_file_path=ngram_file_path,
                                      bigram_file_path=bigram_file_path,
                                      t_limit=30,
                                      search_deepness=1000,
                                      freqs=freqs,
                                      target_score=target_score,
                                      bad_score=bad_score,
                                      print_threshold=print_threshold,
                                      row_bend=1.4,
                                      elem_bend=0.99,
                                      )

    enable_print()
    if value > target_score * 120:
        return hill_encrypt.decrypt(cypher, key, alphabet, freqs), key

    print("Cracking failed. Attempting to guess the length of the key...")

    # Get potential keys
    disable_print()
    guess_table = guess_key_len(cypher, alphabet, freqs=freqs, bigram_file_path=bigram_file_path,
                                ngram_file_path=ngram_file_path, t_limit=60)
    enable_print()

    best_keys = [row[0] for row in guess_table]
    print(f"Keys to check:")
    print("============================================")
    for k in best_keys:
        print(k)
        print("============================================")

    cracked_key = None

    improved = []
    # try to improve pre-guessed keys
    for potential_key in best_keys:
        key_len = potential_key.shape[0]
        print(f"Started test for the key: \n{potential_key}")
        print("Creating subprocesses...")
        cracked_key, cracked_key_value = shotgun_hillclimbing(cypher, key_len, alphabet,
                                                              ngram_file_path=ngram_file_path,
                                                              freqs=freqs,
                                                              bigram_file_path=bigram_file_path,
                                                              t_limit=60 * 40,
                                                              target_score=-3.7,
                                                              bad_score=bad_score,
                                                              print_threshold=print_threshold,
                                                              start_key=potential_key,
                                                              search_deepness=search_deepness,
                                                              row_bend=1.7 ** (key_len - 2),
                                                              elem_bend=1.2 ** (key_len - 2) - 0.2,
                                                              sound_thresholds=[5, 4.5, 4],
                                                              sound=False)

        improved.append((cracked_key_value, cracked_key))

        enable_print()
        if cracked_key_value > target_score * len(cypher):
            print(f"The text is cracked.")
            break

    improved.sort(key=itemgetter(0), reverse=True)
    cracked_key = improved[0][1]

    return hill_encrypt.decrypt(cypher, cracked_key, alphabet, freqs), cracked_key


def guess_key_len(text: str,
                  alphabet: str,
                  bigram_file_path: str,
                  ngram_file_path: str,
                  t_limit: int = 60 * 2,
                  search_deepness: int = 1000,
                  freqs: list[float] | None = None,
                  target_score: float = -2.4,
                  bad_score: float = -3.6,
                  row_bend: float = 1.9,
                  elem_bend: float = 0.9,
                  ):
    """
    Function tries to find correct len of key. Keys longer then 4 are not guarantee to return correct len as best.
    :param text: processed text
    :param alphabet: alphabet used
    :param bigram_file_path: str
    :param ngram_file_path: str
    :param t_limit: time in to analyze different len_key
    :param search_deepness: how many interactions to try before switching key in len_key
    :param freqs: frequency of letters
    :return: list sorted by value structured accordingly [[key,value(how good the key is), did it cracked the text?][key,...]...]
    """
    args = [text, alphabet, ngram_file_path, bigram_file_path, t_limit, search_deepness, freqs, target_score, bad_score]

    alphabet_len = len(alphabet)
    it_args = []
    table = []

    len_from, len_to = 3, 5

    for i in range(len_from, len_to + 1):
        it_args.append([i, row_bend, elem_bend])
        row_bend += 0.6
        elem_bend += 0.2

    with WorkerPool(n_jobs=len_to - len_from + 1, shared_objects=args, daemon=True) as pool:
        generator = pool.imap_unordered(guess_len_unwrapper, iterable_of_args=it_args)
        for next_row in generator:
            if next_row[2]:
                pool.terminate()
                return invert_key(next_row[0], alphabet_len), next_row[1]
            table.append(next_row)

    table.sort(key=itemgetter(1), reverse=True)
    return table


def upgrade_key_unwraper(i, c):
    value = upgrade_key(c, *i)
    return value


def upgrade_key(
        key: np.matrix,
        cypher: str,
        alphabet: str,
        scorer: ngram.Ngram_score,
        a: float,
        b: float,
        row_bend: float,
        elem_bend: float,
        bigram_data: dict,

        freqs: list[float] | None,
        iters: int = 100,
        target_score: float = -2.4,
        print_threshold: float = -3.4
):
    """
    function tries to find better key
    :param iters: number of attempts
    :return:  tuple[key, value of key, did it crack the text?, how many better keys were located]
    """
    alphabet_len = len(alphabet)
    key_len = len(key)

    number_of_upgrades = 0

    key_old = key.copy()
    found = False
    value_old = scorer.score(encrypt(cypher, key_old, alphabet, freqs))
    init_smart = True

    n_smarts = 0
    max_smarts = 50

    n_rows = None

    ns_weights = np.concatenate(([1 for _ in range(key_len - 1)], [key_len - 1]))

    smart_threshold = 0.6

    # try to upgrade the key 'iters' times
    for i in range(iters):
        perc = min(max(a * value_old + b, 0.01), 1)
        perc_rows = perc ** row_bend
        perc_elems = perc ** elem_bend

        r = random.random()

        if key_len != 2 and perc < smart_threshold and init_smart:
            most_prob = int(ceil(key_len * perc))
            possible_ns = np.concatenate((np.arange(1, most_prob), np.arange(most_prob + 1, key_len + 1), [most_prob]))
            n_rows = random.choices(possible_ns, ns_weights)

        if key_len == 2 or perc >= smart_threshold or n_smarts >= max_smarts:
            smarts = False
            n_smarts = 0
        else:
            smarts = True

        if smarts and n_smarts < max_smarts:
            key_new, _ = smart_rand_rows(key, cypher, alphabet, bigram_data=bigram_data, freqs=freqs,
                                         init=init_smart, n_rows=n_rows)

            init_smart = False
            n_smarts += 1

        if not smarts:
            if r < 0.8:
                key_new = randomize_rows(key_old, perc_rows=perc_rows, perc_elems=perc_elems,
                                         alphabet_len=alphabet_len)
            elif r < 0.9:
                key_new = randomize_rows(key_old, perc_rows=perc_rows, perc_elems=1,
                                         alphabet_len=alphabet_len)
            elif r < 0.95:
                key_new = swap_rows(key_old)
                for _ in range(key_len * 2):
                    key_new = swap_rows(key_new)
            else:
                key_new = slide_key(key_old, alphabet_len)

            if random.random() < 0.05:
                key_new = slide_key(key_old, alphabet_len, horizontal=True)

        decoded_new = encrypt(cypher, key_new, alphabet, freqs)
        value_new = scorer.score(decoded_new)

        if value_new > value_old:
            value_normalized = value_new
            init_smart = True

            if value_normalized >= print_threshold:
                print(
                    f"i = {i}, decoded: {decoded_new[:25]}, value: {value_new}, perc = {perc:.3f} "
                    f"perc_rows = {perc_rows:.3f}, perc_elems = {perc_elems:.3f} key = \n{key_new}\n")
                number_of_upgrades += 1
            if value_normalized > target_score:
                print(f'BEST: {decoded_new}, key = \n{key_new}')
                found = True
                key_old = key_new
                value_old = value_new
                break

            key_old = key_new
            value_old = value_new

    return key_old, value_old, found, number_of_upgrades


class Notifier:
    def __init__(self, thresholds: list[float], duration: int = 200, freq_th: int = 600, freq_failure: int = 400,
                 freq_success: int = 800):
        self.t_dict = {k: False for k in thresholds}
        self.duration = duration
        self.freq_th = freq_th
        self.freq_failure = freq_failure
        self.freq_success = freq_success

    def update(self, score):
        threshold_passing = None

        for threshold, already_passed in self.t_dict.items():
            if not already_passed and score > threshold:
                threshold_passing = threshold
                break

        if threshold_passing is not None:
            for _ in range(3):
                winsound.Beep(self.freq_th, self.duration)
            self.t_dict[threshold_passing] = True

    def success(self):
        for _ in range(5):
            winsound.Beep(self.freq_success, self.duration)

    def failure(self):
        for _ in range(5):
            winsound.Beep(self.freq_failure, self.duration)


def linear(x1, y1, x2, y2):
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return a, b


def perc_slope_function(bad_score: float, target_score: float) -> tuple[float, float]:
    return linear(bad_score, 1, target_score, 0.01)


def single_process_shotgun(key_len: int,
                           row_bend: float,
                           elem_bend: float,
                           text: str,
                           alphabet: str,
                           ngram_file_path: str,
                           bigram_file_path: str,
                           t_limit: int = 60 * 2,
                           search_deepness: int = 1000,

                           freqs: list[float] | None = None,

                           target_score: float = -2.4,
                           bad_score: float = -3.6,
                           print_threshold: float = -3.4,

                           ) -> tuple[np.matrix, float, bool]:
    """
    shotgun_hill-climbing classical version
    :param text: process text
    :param key_len: suspected len of key
    :param t_limit: amount of time you are wiling to wait in seconds
    :param search_deepness: how hard to try specific hill
    :param freqs: frequency of letters
    :param start_key: key you suspect is good or close
    :param target_score: change if not using english
    :param bad_score: change if not using english
    :param row_bend: bend of learning function for rows
    :param elem_bend: bend of learning function for rows
    :param print_threshold: the minimal current solution quality value determining to print it to the console
    :return: [key used to encrypt, value]
    """
    scorer = ns(ngram_file_path)

    alphabet_len = len(alphabet)

    t0, itr, j, itr_random = time(), 0, 0, 0

    if key_len == 2:
        text = text[:120]

    word_len = len(text)

    target_score *= word_len
    bad_score *= word_len
    print_threshold *= word_len

    a, b = perc_slope_function(bad_score, target_score)

    with open(bigram_file_path, 'r') as file:
        content = file.readlines()
        splitted = np.array([line.replace("\n", "").split(" ") for line in content])
        splitted[:, 1] = normalize([splitted[:, 1]])
        bigram_data = {k: float(v) for k, v in splitted}

    key_old = random_key(key_len, alphabet_len)
    found = False
    value_old = -1_000_000
    n_iters = 0

    max_buffer = 3
    buffer = []
    buffer_weights = [max_buffer - i for i in range(max_buffer)]

    while time() - t0 < t_limit:
        n_iters += 1
        key_old, value_old, found, _ = upgrade_key(key=key_old, cypher=text, alphabet=alphabet, scorer=scorer,
                                                   a=a,
                                                   b=b,
                                                   row_bend=row_bend,
                                                   elem_bend=elem_bend,
                                                   freqs=freqs,
                                                   iters=search_deepness,
                                                   target_score=target_score,
                                                   print_threshold=print_threshold,
                                                   bigram_data=bigram_data
                                                   )
        if found:
            t = time() - t0
            print(f"time: {t:.2f}, iters: {itr}, {t / max(itr, 1):.2f}s/it")
            return invert_key(key_old, alphabet_len), value_old, found
        else:
            buffer_space_available = len(buffer) < max_buffer

            # insert solution to the buffer
            if buffer_space_available:
                buffer.append((value_old, key_old))
            else:
                buffer.sort(key=itemgetter(0), reverse=True)
                buffer[max_buffer - 1] = (value_old, key_old)

            # get random key from buffer or generate a new one
            if not buffer_space_available and random.random() < 0.9:
                key_old = random.choices(population=buffer, weights=buffer_weights, k=1)[0][1]
            else:
                key_old = random_key(key_len, alphabet_len)

    return invert_key(key_old, alphabet_len), value_old, found


def shotgun_hillclimbing(text: str,
                         key_len: int,
                         alphabet: str,
                         ngram_file_path: str,
                         bigram_file_path: str,
                         t_limit: int = 60 * 5,
                         search_deepness: int = 1000,
                         freqs: list[float] | None = None,
                         start_key: np.matrix | None = None,
                         target_score: float = -2.4,
                         bad_score: float = -3.6,
                         row_bend: float = 1,
                         elem_bend: float = 1,
                         sound: bool = False,
                         sound_thresholds=None,
                         print_threshold: float = -3.4
                         ) -> tuple[np.matrix, float]:
    """
     shotgun_hill-climbing, multiprocessor version with analytical function near finish
     function recognize key_len = 2, and do not activities multiprocessing as it would take more time than just
      cracking on one.
    :param text: process text
    :param key_len: suspected len of key
    :param t_limit: amount of time you are wiling to wait in seconds
    :param search_deepness: how hard to try specific hill
    :param freqs: frequency of letters
    :param start_key: key you suspect is good or close
    :param target_score: change if not using english
    :param bad_score: change if not using english
    :param row_bend: bend of learning function for rows
    :param elem_bend: bend of learning function for rows
    :param sound: play a sound at the end of process
    :param sound_thresholds: Quality thresholds, after reaching which the sound notification occurs.
    :param print_threshold: the minimal current solution quality value determining to print it to the console
    :return: [key used to encrypt, value]
    """
    if sound_thresholds is None:
        sound_thresholds = [-3.2, -3, -2.6]

    scorer = ns(ngram_file_path)

    alphabet_len = len(alphabet)

    t0, itr, j, itr_random = time(), 0, 0, 0

    if key_len == 2:
        text = text[:120]

    word_len = len(text)
    sound_thresholds = sound_thresholds * word_len

    target_score *= word_len
    bad_score *= word_len
    print_threshold *= word_len

    a, b = perc_slope_function(bad_score, target_score)

    with open(bigram_file_path, 'r') as file:
        content = file.readlines()
        splitted = np.array([line.replace("\n", "").split(" ") for line in content])
        splitted[:, 1] = normalize([splitted[:, 1]])
        bigram_data = {k: float(v) for k, v in splitted}

    # Create notifier, give a list of thresholds after which the beeping occurs.
    notifier = Notifier(sound_thresholds) if sound else None

    # single process shotgun for key_lem = 2
    if key_len == 2:
        key_old = random_key(key_len, word_len)

        while time() - t0 < t_limit:
            # q, to_change = random.choices(buffer, k=1)[0]

            # print(f"i choose {to_change}, q = {q / 120}")
            key_old, value_old, found, a = upgrade_key(key=key_old, cypher=text, alphabet=alphabet, scorer=scorer,
                                                       a=a,
                                                       b=b,
                                                       row_bend=row_bend,
                                                       elem_bend=elem_bend,
                                                       freqs=freqs,
                                                       iters=search_deepness,
                                                       target_score=target_score,
                                                       print_threshold=print_threshold,
                                                       bigram_data=bigram_data
                                                       )
            if found:
                if notifier is not None:
                    notifier.success()
                t = time() - t0
                print(f"time: {t:.2f}, iters: {itr}, {t / max(itr, 1):.2f}s/it")
                return invert_key(key_old, alphabet_len), value_old
            else:
                key_old = random_key(key_len, alphabet_len)

    # for other lengths launch multiprocessing
    else:
        args = [text, alphabet, scorer, a, b, row_bend, elem_bend, bigram_data, freqs, search_deepness,
                target_score,
                print_threshold]

        key_old = random_key(key_len=key_len, alphabet_len=alphabet_len)
        it_args = [random_key(key_len, alphabet_len) for _ in range(key_len * 12)]

        if start_key is not None:
            it_args.pop()
            it_args.append(start_key)

        value_old = scorer.score(encrypt(text, key_old, alphabet, freqs))

        with WorkerPool(n_jobs=12, shared_objects=args, keep_alive=True, daemon=True) as pool:
            while time() - t0 < t_limit:
                generator = pool.imap_unordered(upgrade_key_unwraper, iterable_of_args=it_args)

                table = []
                for next_row in generator:
                    if next_row[2]:
                        if sound:
                            notifier.success()
                        t = time() - t0
                        print(f"time: {t:.2f}, iters: {itr}, {t / itr:.2f}s/it")
                        pool.terminate()

                        return invert_key(next_row[0], alphabet_len), next_row[1]
                    table.append(next_row)
                itr += 1
                # table.sort(key=lambda row: (row[1]), reverse=True)
                table.sort(key=itemgetter(1), reverse=True)
                key_old = table[0][0]
                value_old = table[0][1]

                # Update the notifier
                if notifier is not None:
                    notifier.update(value_old)

                if table[0][2]:
                    if sound:
                        notifier.success()
                    t = time() - t0

                    print(f"time: {t:.2f}, iters: {itr}, {t / max(itr, 1):.2f}s/it")
                    return invert_key(key_old, alphabet_len), value_old

                total = sum([row[3] for row in table])
                print(f"Number of upgrades: {total}")
                if total < 5 and itr_random < 20:
                    to_len = len(it_args)
                    itr_random += 1
                    it_args = [row[0] for row in table[:int(ceil(len(table) / (2 - itr_random / 20)))]]
                    for i in range(len(it_args), to_len):
                        it_args.append(random_key(key_len, alphabet_len))
                else:
                    it_args = [row[0] for row in table[:int(ceil(len(table) / 2))]] * 2
                    it_args.pop()
                    it_args.append(random_key(key_len, alphabet_len))
                print(f"Process Iteration of size: {len(it_args)}, iteration of tasks nr: {itr}")

                itr += 1

    t = time() - t0
    print(f"time: {t:.2f}, iters: {itr}, {t / max(itr, 1):.2f}s/t")

    if notifier is not None:
        notifier.failure()

    return invert_key(key_old, alphabet_len), value_old


def get_scores(text: str, alphabet: str, scorer: ngram.Ngram_score):
    text = utils.preprocess_text(text, alphabet)
    text_len = len(text)

    # generate target score
    target_score = scorer.score(text) / text_len

    # generate bad score
    key = random_key(3, len(alphabet))
    enc = encrypt(text, key, alphabet)
    bad_score = scorer.score(enc) / text_len

    return target_score, bad_score
