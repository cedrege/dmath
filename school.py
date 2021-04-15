#import numpy as np
"""
class dmath():
    
    @staticmethod
    def binomialdist(n,k):
        return np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k))


    #binomialverteilung
    #mit zurücklegen
    #k = anz. erfolge n = anz. ziehungen p = chance für erfolg
    @staticmethod
    def binomial(k,n,p):
        return dmath.binomialdist(n,k) * p**k * (1-p)**(n-k)
    
    #hypergeometrische verteilung
    #ohne zurücklegen
    #N = anz. möglichkeiten M = anz. guter möglichkeiten n = anz. ziehungen k = anz. erfolge
    @staticmethod
    def hypergeo(N,M,n,k):
        return dmath.binomialdist(M,k) * dmath.binomialdist(N - M, n - k) / dmath.binomialdist(N,n)
    
    #poisson verteilung
    #abschätzung
    #n = anz. möglichkeiten p = chance für erfolg k = anz. erfolge
    @staticmethod
    def poisson(n,p,k):
        return (n*p)**k / np.math.factorial(k) * np.math.exp(-n*p) 
"""



# Documentationstyleguide: https://google.github.io/styleguide/pyguide.html
from pandas import DataFrame
from numpy import cumsum, e


def fac(n: int) -> int:
    return n * fac(n-1) if n > 1 else 1

def n_tief_r_bad(n: int, r: int) -> float:
    """deprecated - Use binomial_coefficient() instead
    """
    return fac(n) / (fac(r) * fac(n - r))

def binomial_coefficient(n: int, k: int) -> float:
    """TODO:

    Args:
      n: 
      k: 

    Raises:
      ValueError: When divisor is zero

    Returns:
      The binomial coefficient of the provided parameters
    """
    dividend = 1
    divisor = 1

    for i in range(n, n-k, -1):
        dividend *= i
        divisor  *= ((n - i) + 1)
    
    if divisor == 0:
        raise ValueError("Divisor is zero! Check parameters")

    return dividend / divisor

def binomial_distribution(n: int, k: int, p: int) -> float:
    """Calculates the binomial distribution for the given paramters

    Args: 
      n: number of trials
      k: number of times for a specific outcome within n trials
      p: probability of failure on a single trial

    Returns:
      The binomial distribution for the given paramters
    """
    return binomial_coefficient(n, k) * p**k * (1-p)**(n-k)

def cumsum_binomial_distribution(n: int, p: int, sum_range: tuple) -> float:
    """Calculates the cumulative sum of the binomial distribution between a given range

    Args:
      n: number of trials
      p: probability of failure on a single trial
      sum_range: chance accumulated between the given range.
                 The +1 for the upper bound of the range function will be automatically added.
                 So, do not add it yourselves!

    Returns:
      The cumulative sum of the binomial distribution between a given range
    """
    sum = 0
    for i in range(sum_range[0], sum_range[1] + 1):
        sum += binomial_distribution(n, i, p)
    return sum

def poisson_distribution(my: int, k: int) -> float:
    """ Calculates the poisson distribution for a given my and k
    
    Args:
      my: n * p
      k: number of times for a specific outcome within n trials

    Returns:
      The poisson distribution for the given paramters
    """
    return my**k/fac(k) * e**(-my)

def cumsum_poisson_distribution(my: int,sum_range: tuple) -> float:
    """ Calculates the cumulative sum of the poisson distribution for a given my
    
    Args:
      my: n * p
      sum_range: chance accumulated between the given range.
                 The +1 for the upper bound of the range function will be automatically added.
                 So, do not add it yourselves!

    Returns:
      The poisson distribution for the given paramters
    """
    sum = 0
    for i in range(sum_range[0], sum_range[1] + 1):
        sum += poisson_distribution(my, i)
    return sum

def ewfin(n: int, p: float):
    complement = 1 - p
    arr = [p]
    for i in range(1, n):
        arr.append((complement)**(i) * p)

    arr.append((complement)**n)

    arr2 = []
    for i, v in enumerate(arr[:-1]):
        arr2.append((i+1) * v)

    arr2.append(arr[-1] * (len(arr) -1))

    tl = [[str(x) for x in range(n+1)], arr2, cumsum(arr2)]
    print(DataFrame(tl, index=["Xi", "P(x = Xi)", "sum(P(x))"], columns=[str(" ") for x in range(n+1)]))
    print(f"\nErwartungswert: {sum(arr2)}")
