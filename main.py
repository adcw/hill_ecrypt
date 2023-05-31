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

Tematem projektu jest łamanie szyfru Hilla - szyfrowaniu opierającym się na mnożeniu tekstu jawnego przez klucz będący
macierzą i odszyfrowywaniu polegającym na mnożeniu kryptotekstu przez odwrotność klucza.

Zakładamy, że nie jest znana długość klucza. Proces łamania szyfru składa się z następujących etapów:

1. Sprawdzenie, czy da się odszyfrować tekst kluczami o wymiarach 2x2

    W tym celu korzystamy z klasycznej metody shotgun hillclimbing, lekko zmodyfikowanej na potrzeby zadania.
    Pierwszym etapem jest wygenerowanie wielu losowych kluczy i wrzucenie ich do bufora.
    Następnie za każym razem losowany jest klucz z bufora i podejmuje się próbę ulepszenia klucza.
    Im lepszy klucz, tym większe prawdopodobieństwo, że zostanie wylosowany.
    Po ulepszeniu nowy klucz trafia spowrotem do bufora zastępując najgorszy klucz.
    Prócz tego występuje mała szansa na to, że zamiast losować klucz z bufora wygenerujemy nowy klucz.
    
2. Próba odgadnięcia długości klucza

    W tym celu uruchamiamy algorytm opisany wyżej, lecz dla długości klucza od 3 do 6 (wartość tę można zmienić).
    Każdy wariant długości klucza uruchamia się na osobnym prccesie, dzięki czemu skracany jest czas
    wykonania wszystkich sprawdzań. Pozwala to na lekkie przedłużenie każdego z procesów, dzięki czemu ocena długości
    klucza ma wyższą dokładność.

3. Iterowanie po najbardziej prawdopodobnych długościach klucza i dokładna próba łamania szyfru.

    W tym celu uruchamiana jest druga wersja shotgun hillclimbing, która skupia się na jednej długości klucza
    i dzieli proces wspinaczki na wiele procesów, maksymalizując użycie dostępnych zasobów komputera.
    Taki zabieg pozwala na stosunkowo szybkie łamanie szyfrów długości 3 oraz 4, teoretycznie i dłuższych.
    
    
W projekcie zastosowano kilka sztuczek pozwalających na optymalizację łamania szyfru:
- procent zmiany klucza maleje wraz ze wzrostem jakości odszyfrowywanego nim tekstu, dzięki czemu na samym początku
  klucz jest zmieniany w stopniu znacznym, a pod koniec tylko nieznacznie.
  
- jeśli jakość klucza osiągnie 50% docelowej jakości, uruchamiana jest funkcja, która ocenia, które wiersze
  należy zmieniać. Bierze się to stąd, że po przekroczeniu tej granicy generowany klucz często posiada połowę
  wierszy identyczną co klucz łamany, co oznacza, że powinniśmy się skupić jedynie na wierszach różniących się.
  Wybór wierszy do zmieniania polega na obliczeniu jakości bigramów kryptotekstu, a następnie rzutowaniu tej jakości
  na pojedyńcze litery, dzięki czemu wiadomo, które litery wydają się być odstające. Następnie okresowo sumujemy
  jakości dla każdej litery, otrzymując listę wartości o długości równej długości klucza. Wysokie wartości tej sumy
  oznaczają, że odpowiadający tej sumie wiersz generowanego klucza tworzy w kryptotekście znaki wyraźnie odstające,
  czyli należy zmieniać ten wiersz. Metoda ta w 95% przypadków poprawnie wskazuje wiersz, który różni się
   od wiersza docelowego
   
- w związku z tym, że często generowany klucz posiada prawidłowe wiersze, lecz ułożone w złej kolejności,
  z małą szansą zamienia się w sposób losowy kolejność wierszy klucza. Zamiana dwóch wierszy klucza wykonywana jest
  ilość razy proporcjonalną do długości klucza.
"""


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
    5       | 4        | 1.5       | 
    2       | 1.3      | 0.8       |
    
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

    key_len = 3
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