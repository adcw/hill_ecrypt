def gcd(a, b):
    """
    Oblicza największy wspólny dzielnik dwóch liczb całkowitych.
    """
    while b:
        a, b = b, a % b
    return a


def are_coprime(a, b):
    """
    Sprawdza, czy dwie liczby są względnie pierwsze.
    """
    return gcd(a, b) == 1
