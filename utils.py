from numpy import linalg, matrix, floor, ceil, round
from math import gcd


# def gcd(a, b):
#     """
#     Oblicza największy wspólny dzielnik dwóch liczb całkowitych.
#     """
#     while b:
#         a, b = b, a % b
#     return a


def are_coprime(a, b):
    """
    Checks if two values are coprime.
    """
    return gcd(a, b) == 1


def mod_inverse_matrix(m: matrix, modulo: int) -> matrix | None:
    """
    Modulo inverse of an matrix
    :param m: the matrix
    :param modulo: the modulo
    :return: a matrix inversion
    """
    det = round(linalg.det(m))

    if gcd(int(det), modulo) != 1:
        return None

    m_inv = linalg.inv(m)
    m_inv_modulo = (m_inv * det * pow(int(det), -1, modulo)) % modulo
    m_int = round(m_inv_modulo).astype(int)
    return m_int
