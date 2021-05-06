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
from IPython.display import display, Math
from functools import reduce
from math import log 


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


def check_prime(x):
    """Takes in a number and Returns True if it's a Prime number, otherwise returns False"""
    # Python program to check if
    # given number is prime or not

    num = x

    # If given number is greater than 1
    if num > 1:

        # Iterate from 2 to n / 2
        for i in range(2, int(num/2)+1):

            # If num is divisible by any number between
            # 2 and n / 2, it is not prime
            if (num % i) == 0:
                return False
                break
        else:
            return True

    else:
        return False

def small_fermat(a:int,b:int,x:int, steps=False):
    """implementierung für step by step Anleitung des kleinen Fermats
    
    Args:
      a: Basis
      b: Exponent
      x: zu welchem modulo
      steps: Wenn True, zeigt Schritte auf
      
    Returns:
      a^b mod x
      
    """
    if check_prime(x):
        new_exp = b // (x-1)
        exp_rest = b % (x-1)
        display(Math(f'{a}^{{{b}}}\ mod\ {x}'))
        if steps and exp_rest > 0:
            display(Math(f'(\f{a}^{{{x-1}}})^{{{new_exp}}}\ *\ {a}^{{{exp_rest}}} \ mod\ {x}'))
            display(Math(f'(\f{a}^{{{x-1}}}\ mod\ {x})^{{{new_exp}}}\ *\ {a}^{{{exp_rest}}}\ mod\ {x}'))
            display(Math('Kleiner\ Satz\ von\ Fermat\ =>\ m^{(p-1)}\ mod\ p\ =\ 1\ ||\ wenn\ p\ =\ prime'))
            display(Math(f'1^{{{new_exp}}}\ mod\ {x}'))
            display(Math(f'{a**exp_rest}\ mod\ {x}'))
        if steps and exp_rest == 0:
            display(Math(f'(\f{a}^{{{x-1}}})^{{{new_exp}}}\ mod\ {x}'))
            display(Math(f'(\f{a}^{{{x-1}}}\ mod\ {x})^{{{new_exp}}}\ mod\ {x}'))
            display(Math('Kleiner\ Satz\ von\ Fermat\ =>\ m^{(p-1)}\ mod\ p\ =\ 1\ ||\ wenn\ p\ =\ prime'))
            display(Math(f'1^{{{new_exp}}}\ mod\ {x}'))
            display(Math(f'{a**exp_rest}\ mod\ {x}'))
        if a**exp_rest > x:
            display(Math(f'{a**exp_rest%x}\ mod\ {x}'))
        else:
            display(Math(f'{a**exp_rest%x}\ mod\ {x}'))
        
    else:
        print('Kleiner Fermat funktioniert nur wenn mod einer Primzahl')
        display(Math(f'{a}^{{{b}}}\ mod\ {x}'))
        display(Math(f'{a**b%x}\ mod\ {x}'))

def euler_prime(x:int,y=0):
    """Berechnung, wieviele Primzahlen in einem Bereich anzutreffen sind
    
    Args:
      x: Ende des Intervalls
      y: Anfang des intervalls, von 0 her falls nicht angegeben
    Returns:
      erwartete Primzahlen in bereich zwischen x und y
      
    """
    display(Math('Formel\ für\ ungefähre\ Bestimmung:\ \\frac{x}{ln(x)}\ -\ \\frac{y}{ln(y)}'))
    if y !=0:
        result = x/(log(x)) - y/(log(y))
        if str(result).count("e") > 0:
            str_1 = str(result).split("+")
            str_1[1] = f"{{{str_1[1]}}}"
            display(Math("^".join(str_1)))
        else:
            display(Math(str(result)))
    else:
        display(Math(f"{{{x/log(x)}}}"))

def bayes(fpr, fnr, verb): #oder 1-spezifität, 1-sensitivität, verb
    """Implementierung von Bayes rule
    
    Args:
      fpr: False positive rate
      fnr: False negative rate
      verb: Verbreitung
      
    Returns:
      alle möglichen chancen nach erfolgtem Test"""
    sensitivity = 1-fnr
    spezifitaet = 1-fpr

    bayes_pos = sensitivity * verb / (sensitivity * verb + fpr * (1-verb))
    bayes_neg = spezifitaet * (1-verb) / (spezifitaet * (1-verb) + fnr * verb)
    abs_pos = verb * sensitivity + (1-verb) * fpr
    abs_neg = verb * fnr + (1-verb) * spezifitaet
    print("chance nach pos. test wenn wirklich pos:", bayes_pos)
    print("chance nach neg. test wenn wirklich neg.:", bayes_neg)
    print("chance nach pos. test wenn NICHT pos:", 1-bayes_pos)
    print("chance nach neg. test wenn NICHT neg.:", 1-bayes_neg)
    print("abs. wahrscheinlichkeit für pos. Ereignis:", abs_pos)
    print("abs. wahrscheinlichkeit für neg. Ereignis:", abs_neg)

def sma(n, p, m, steps=False):
    """Implementierung von Square and Multiply
    
    Args:
      n: Basis
      p: potenz bzw exponent
      m: modulo
      steps: wenn True zeigt es Schritte an
    Returns:
      n^p%m"""
    binary = bin(p)
    binary = binary[2:]
    binary_short = str(binary[1:])
    todo = []
    new = n
    count = 1
    a = ""
    b = f"{n}"
    for i in binary_short:
        if i == "1":
            todo.append("QM")
        if i == "0":
            todo.append("Q")
    for i in binary:
        if i == "1":
            a += f"2^{(len(binary)-count)}+"
        count += 1
    for i in todo:
        if i == "QM":
            new = (new**2)*n
            b += f"\ Q\& M\ mod\ {m}\ => {new%m}\ "
        if i == "Q":
            new = new**2
            b += f"\ Q\ mod\ {m}\ => {new%m}\ "
    if steps:
        display(Math(f'{n}^{{{p}}}\ mod\ {m}'))
        display(Math(f'{p}\ =\ {binary}\ in\ binary'))
        display(Math(f'nun\ gilt\ für\ jedes\ 1\ QM\ und\ 0\ Q\ und\ das\ erste\ 1\ wird\ ignoriert\ =>\ {binary[1:]}\ wird\ zu\ {todo}'))
        display(Math(f'Q\ =\ hoch\ 2\ ;\ M\ =\ multiplizieren\ mit\ {n}'))
        display(Math(f'{n}^{{{a[:-1]}}}\ mod\ {m}'))
        display(Math(f'{b}'))
        display(Math(f'{new%m}\ mod\ {m}'))
    return new%m
