"""
#import numpy as np
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
from typing import get_type_hints


def strict_types(function):
    def type_checker(*args, **kwargs):

        # Get dict of (arg. name, expected arg. type)
        # get_type_hints is often the same as "function.__annotations__" (see: docs.python.org/3/library/typing.html)
        hints = get_type_hints(function)

        # Get dict of (arg. name, given arg.)
        all_args = kwargs.copy()
        all_args.update(dict(zip(function.__code__.co_varnames, args)))

        # Check if given arg. type is the same as expected arg. type
        for argument, argument_type in ((i, type(j)) for i, j in all_args.items()):
            if argument in hints:
                if not issubclass(argument_type, hints[argument]):
                    raise TypeError('Type of {} is {} and not {}'.format(argument, argument_type, hints[argument]))

        result = function(*args, **kwargs)

        if 'return' in hints:
            if type(result) != hints['return']:
                raise TypeError('Type of result is {} and not {}'.format(type(result), hints['return']))

        return result

    return type_checker


def fac(n: int) -> int:
    return n * fac(n-1) if n > 1 else 1


def binomial_coefficient(n: int, k: int) -> float:
    """ Berechnet den Binomialkoeffizienten zweier Zahlen.
        Die Paremeter wurden so implementiert, dass man n tief k lesen wuerde.
        Somit ergibt sich die Aussage: Man kann k verschiedene Objekte von der Menge n auswaehlen kann.

    Args:
      n: Anzahl Moeglichkeiten
      k: Anzahl Erfolge

    Raises:
      ValueError: Wenn dividend null ist

    Returns:
      Binomialkoeffizient, berechnet aus den angegeben Parametern
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
    """ Berechnet die Binomialverteilung anhand der angegeben Werte.

    Args: 
      n: Anzahl Moeglichkeiten
      k: Anzahl Erfolge
      p: Wahrscheinlichkeit auf Erfolg bei einem Versuch

    Returns:
      Binomialverteilung, berechnet aus den angegeben Parametern
    """
    return binomial_coefficient(n, k) * p**k * (1-p)**(n-k)


def cumsum_binomial_distribution(n: int, p: int, sum_range: tuple) -> float:
    """ Berechnet die Binomialverteilung anhand der angegeben Werte.
        Die Wahrscheinlichkeiten werden in einem bestimmten Bereich gerechnet und zusammensummiert.

    Args:
      n: Anzahl Moeglichkeiten
      p: Wahrscheinlichkeit auf Erfolg bei einem Versuch
      sum_range: Bereich, indem die Wahrscheinlichkeiten zusammengerechnet werden.
                 Es wird automatisch +1 fuer die obere Grenze hinzugefuegt.
                 Manuelles hinzufügen wird in einem falschen Resultat resultieren.

    Returns:
      Summierte Resultate der Binomialverteilung in einem bestimmten Bereich
    """
    sum = 0
    for i in range(sum_range[0], sum_range[1] + 1):
        sum += binomial_distribution(n, i, p)
    return sum


def poisson_distribution(my: int, k: int) -> float:
    """ Berechnet die Poisson Verteilung fuer my und k
    
    Args:
      my: Anzahl Moeglichkeiten mal potenzielle Chance (n * p)
      k: Anzahl Erfolge

    Returns:
      Resultat der Poisson Verteilung fuer die angegeben Parameter
    """
    return my**k/fac(k) * e**(-my)


def cumsum_poisson_distribution(my: int, sum_range: tuple) -> float:
    """ Berechnet die Poisson Verteilung fuer my und k.
        Die Wahrscheinlichkeiten werden in einem bestimmten Bereich gerechnet und zusammensummiert.
    
    Args:
      my: Anzahl Moeglichkeiten mal potenzielle Chance (n * p)
      sum_range: Bereich, indem die Wahrscheinlichkeiten zusammengerechnet werden.
                 Es wird automatisch +1 fuer die obere Grenze hinzugefuegt.
                 Manuelles hinzufügen wird in einem falschen Resultat resultieren.

    Returns:
      Summierte Resultate der Poisson Verteilung ueber den angegeben Bereich
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


def prime_facs(n: int, steps=False) -> list:
    """ Berechnet die Primfaktorzerlegung und zeigt die Schritte auf

    Args:
      n: Die Zahl welche Primfaktorzerlegt werden soll
      steps: Wenn True, zeigt Schritte auf
    
    Returns:
      Liste mit einzelnen Primfaktorwerten
    """
    list_of_factors=[]
    i=2
    while n>1:
        if n%i==0:
            if steps:
                display(Math(f"{n} \div {i} = {n // i}"))
            list_of_factors.append(i)
            n=n//i
            i=i-1
        i+=1  
    return list_of_factors


def euclid_ggt_print(a, b, steps = False, calc = 1):
    """ Do not use this function! USE euclid_ggt instead.
    """
    if a == 0: return b
    if b == 0: return a

    m = a % b

    if steps:
        display(Math(f"{a} = {calc} \cdot {b} + {m}"))
    
    return euclid_ggt_print(b, m, steps, b // (m) if m > 0 else 1)


def euclid_ggt(a: int, b: int, steps = False) -> int:
    """ Berechnet den groessten gemeinsamen Teiler zweier Inteager
    
    Args:
      a: Erste Inteager Zahl
      b: Zweite Inteager Zahl
      steps: Wenn True, zeigt Schritte auf

    Returns:
      Groesster gemeinsamer Teiler von a und b
    """
    return euclid_ggt_print(a, b, steps)


def euclid_ggt_extended_print(a: int, b: int, steps = False):
    """ Do not use this function! USE euclid_ggt_extended instead.
    """
    if b == 0: return (a, 1, 0)
    if a == 0: return (b, 1, 0)

    d1, s1, t1 = euclid_ggt_extended_print(b, a % b, steps)
    g = (d1, t1, s1 - (a // b)* t1)
    if steps:
        display(Math(f"{d1} = {g[2]} \cdot {b} {f'-{abs(t1)}' if t1 < 0 else f'+{t1}'} \cdot {a}"))

    return g 


def euclid_ggt_extended(a: int, b: int, steps = False) -> int:
    """ Berechnet die inversen Elemente zweier Inteager Zahlen.
        Die Zahlen müssen Teilerfremd sein.
    
    Args:
      a: Erste Inteager Zahl
      b: Zweite Inteager Zahl
      steps: Wenn True, zeigt Schritte auf

    Returns:
      Inverses Element am Index 0 des Tupels
    """
    if euclid_ggt(a, b) != 1:
        raise ValueError("Extended euclid requires two numbers which are coprime!")

    d, s, t = euclid_ggt_extended_print(a, b, steps)
    s_1 = s
    t_1 = t
    if s < 0: s_1 = s + b
    if t < 0: t_1 = t + a
    if steps:
        print()
        display(Math(f"{a}^{{-1}}\mod {b} = {s_1}"))
        display(Math(f"{b}^{{-1}}\mod {a} = {t_1}"))
    return (s_1, t_1)


def euler_phi(n: int, steps=False, primfac_steps=False) -> int:
    """ Implementation der Eulerischen Phi Funktion mit steps. Zeigt nur die Anzahl der Elemente.
        Wenn die Menge der Zahlen auch wichtig ist "euler_phi_set" nutzen!
    
    Args:
      n: Inteager Zahl
      steps: Wenn True, zeigt Schritte auf
      primfac_steps: Wenn True, zeigt zusaetzlich Schritte fuer Primfaktorzerlegung an

    Returns:
      Anzahl Element der Eulerischen Phi Funktion fuer n
    """
    prime_factors = prime_facs(n, primfac_steps)

    if n < 2:
        if steps:
            display(Math(f"\Phi(n < 2) = 1"))
        return 1

    if len(prime_factors) == 1:
        if steps:
            display(Math(f"\Phi(p) = p - 1 \Rightarrow \Phi({n}) = {n - 1}"))
        return n - 1
    else:
        last_value = 0
        base_value = 1
        stepsstr = []
        for i in sorted(prime_factors):
            if i == last_value:
                continue
            else:
                last_value = i
                base_value *= (i - 1) * i**(prime_factors.count(i) - 1)
                if steps:
                    stepsstr.append(f"({i - 1}) \cdot {i}^{{{prime_factors.count(i) - 1}}}")
        
        stepsstr_n = '\cdot'.join(stepsstr)
        display(Math(f"\Phi(n) = {stepsstr_n} = {base_value}"))

    return base_value


def euler_phi_set(n: int, steps=False, primfac_steps=False) -> int:
    """ Implementation der Eulerischen Phi Funktion mit steps, welche auch die Mengen anzeigt.
        Nicht fuer grosse Zahlen nutzen!
        Fuer grosse Zahlen und wenn nur das Resultat wichtig ist "euler_phi" nutzen!
    
    Args:
      n: Inteager Zahl
      steps: Wenn True, zeigt Schritte auf
      primfac_steps: Wenn True, zeigt zusaetzlich Schritte fuer Primfaktorzerlegung an

    Returns:
      Die Zahlenmenge der Eulerischen Phi Funktion
    """
    prime_factors = prime_facs(n, primfac_steps)

    if n < 2:
        difference_all_tmp = {1}
    else:
        # convert to set to remove duplicates
        tmp = []
        for i in range(1, n):
            for j in set(prime_factors):
                if i % j == 0:
                    tmp.append(i)
        
        difference_all_tmp = sorted({x for x in range(1, n)} - set(tmp))
    if steps:
        display(Math(f"\mathbb{{Z}}_{{{str(n)}}}^* = \{{{ ', '.join(str(x) for x in difference_all_tmp) } \}} \Rightarrow \lvert \mathbb{{Z}}_{{{str(n)}}}^* \\rvert = \Phi({str(n)}) = {str(len(difference_all_tmp))} "))

    return set(difference_all_tmp)


def calc_inv_based_on_euler_phi_set(n: int, steps=False, euclid_ext_steps=False) -> int:
    """ Berechnet alle invertiebaren Elemente im subset 0 - n der euler_phi Methode.
        Elemente die nicht ausgegeben werden sind nicht invertierbar.

    Args:
      n: Bereich der Euler Phi Funktion
      steps: Wenn True, zeigt Schritte auf
      euclid_ext_steps: Zeigt zudem die Berechnung des erweiterten Euklids an

    Returns:
      Liste mit Tupels, index = 0 Position im Bereich, index 1 = invertiertes Element davon.
    """
    eu_phi_set = euler_phi_set(n, steps)

    subset = []
    for i in eu_phi_set:
        subset.append((i, euclid_ggt_extended(i, n, euclid_ext_steps)[0]))

    if steps:
        for j in subset:
            display(Math(f"{j[0]}^{{-1}}={j[1]}"))

    return subset


def crt(a: tuple, b: tuple, *args: tuple, steps=False) -> int:
    """ Implementation der Chinesischen Restwerts mit steps. Diese Funktion löst ein
        x für 2 oder mehrere Modulo Operationen auf.
        Beispiel:
          x === 3 mod 5
          x === 2 mod 7
          x === 4 mod 9
          , wobei die erste Zahl der Restwert und die zweite der Divisor ist.

          x ist somit 58
    
    Args:
      a: Erstes, benoetigtes modulo Tuple z. B: (3 mod 5 = (3,5)) 
      b: Zweites, benoetigtes modulo Tuple z. B: (2 mod 7 = (2,7)) 
      *args: Weitere, optionale Tuple
      steps: Wenn True, zeigt Schritte auf

    Returns:
      Nach x aufgeloeste Zahl abhaengig von den Parametern
    """
    # check if all params are tuples
    if type(a) != tuple: raise ValueError("Only tuples allowed")
    if type(b) != tuple: raise ValueError("Only tuples allowed")
    for i in args:
        if type(i) != tuple: raise ValueError("Only tuples allowed")

    #build new big boy tuple
    all_divisors = ((a[1]),) + ((b[1]),) + tuple((ar[1] for ar in args))
    all_remainder = ((a[0]),) + ((b[0]),) + tuple((ar[0] for ar in args))

    # check if all values are coprime
    for i in range(len(all_divisors)):
        if i == len(all_divisors) - 1:
            if euclid_ggt(all_divisors[i], all_divisors[0]) != 1:
                #print(all_divisors[i], all_divisors[0])
                raise ValueError("Entered values are not coprime!")
        else:
            if euclid_ggt(all_divisors[i], all_divisors[i+1]) != 1:
                raise ValueError("Entered values are not coprime!")
    
    # Calculate M and m
    m = reduce(lambda x,y: x * y, all_divisors)
    M = [m // m_x for m_x in all_divisors]

    if steps:
        display(Math(f"m = {{{' * '.join(str(x) for x in all_divisors)}}} = {m}"))
        for c, v in enumerate(zip(M, all_divisors)):
           display(Math(f"M_{c+1} = \\frac{{m}}{{m_{c+1}}} = \\frac{{{m}}}{{{v[1]}}} = {v[0]}"))

    # Calculate y_i
    y = []
    for c, i in enumerate(zip(M, all_divisors)):
        for j in range(1, i[1]+1):
            if (j * i[0]) % i[1] == 1:
                y.append(j)
                if steps:
                    display(Math(f"{i[0]} \cdot {j} \equiv {j * i[0]} \equiv 1 \mod{i[1]} \\Rightarrow y_{c+1} = {j} \equiv {i[0]}^{{-1}}"))
                break
    
    # Calculate solution
    crt_sum = 0
    stepsstr = []
    for i in zip(all_remainder, M, y):
        crt_sum += i[0] * i[1] * i[2]
        if steps:
            stepsstr.append(f"{i[0]} \cdot {i[1]} \cdot {i[2]}")
  
    res = crt_sum % m

    if steps:
        display(Math(f"x = \sum_{{i=1}}^{len(stepsstr)} r_i \cdot M_i \cdot y_i  = {' + '.join(stepsstr)} = {crt_sum} \equiv {res} \mod{m}"))
        display(Math("All \ solutions"))
        display(Math(f"{res} + k * {m}"))
    return res


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


def small_fermat(a: int,b: int,x: int, steps=False):
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


def euler_prime(x: int, y=0):
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


@strict_types
def check_primitive_element(p: int, s: int, steps=False) -> bool:
    """ Diese Funktion ueberprueft ob der Integer s ein primitives Element
        von der Primzahl p ist. Dies ist der Fall wenn s^i mod p (0 < i < p)
        alle Elemente in Z_p erzeugt.

    Args:
        p: Primzahl
        s: Integer von dem erwartet wird, ein primitives Element von p zu sein
        steps: Flag, wenn True werden die Schritte in Latex ausgegeben

    Raises:
        ValueError: Wenn p keine Primzahl ist

    Returns:
        Boolean der True ist, falls s ein erzeugendes/primitives Element von p ist.
    """

    if not check_prime(p):
        raise ValueError(f'{p} is not a prime number!')

    Z = [i for i in range(1, p)]

    if steps:
        display(Math('\mathrm{Input:}'))
        display(Math(f'p = {p}, s = {s}'))
        display(Math(f'\mathbb{{Z}}_{{{p}}}^{{*}} = {Z}'))
        print()
        display(Math('\mathrm{Berechnung:}'))

    sequence = [s]

    for n in range(1, p):

        if steps:
            display(Math(f'{sequence[-1]} \cdot {s} \;\bmod\; {p} = {sequence[-1] * s % p}'))

        sequence.append(sequence[-1] * s % p)

    sequence = list((set(sorted(sequence))))

    if sequence == Z:
        return True
    else:
        return False


@strict_types
def get_primitive_elements(p: int) -> list:
    """ Diese Funktion berechnet mithilfe der Funktion "check_primitive_element"
        eine Liste der primitiven Elemente von "p".

    Args:
        p: Primzahl

    Raises:
        ValueError: Wenn p keine Primzahl ist

    Returns:
        Liste der primitiven Elemente. Leere Liste wenn keine primitiven Elemente
        gefunden werden.
    """

    if not check_prime(p):
        raise ValueError(f'{p} is not a prime number!')

    out = []
    for nr in range(2, p):
        if check_primitive_element(p, nr):
            out.append(nr)

    return out


@strict_types
def diffie_hellman(p: int, s: int, a: int, b: int, steps=False) -> tuple:
    """ Implementation der Diffie-Hellman Verschluesselung. Beteiligt sind die
        zwei Parteien A und B, welche sich gegenseitig verschluesselte
        Nachrichten zusenden wollen.

    Args:
      p: Primzahl, 1. Teil des Schluessels
      s: Primitives Element von p, 2. Teil des Schluessels
      a: Zufaellig ausgewaehlte Zahl von Partei A welche kleiner als p ist
      b: Zufaellig ausgewaehlte Zahl von Partei B welche kleiner als p ist
      steps: Flag, wenn True werden die Schritte in Latex ausgegeben

    Raises:
      ValueError: Wenn p keine Primzahl und/oder s kein primitives Element von p ist.

    Returns:
      Die Funktion liefert den Wert "alpha" von A, den Wert "beta" von B und den
      gemeinsamen Schluessel, welcher jeweils von A mit beta und von B mit alpha errechnet wird.
    """

    # check if a and b are smaller than p
    if not a < p or not b < p:
        raise ValueError(f'a and b must be smaller than p! a={a}, b={b}, p={p}')

    # check if p is a prime number
    if not check_prime(p):
        raise ValueError(f'{p} is not a prime number!')

    # check if s is a element of Zp* (primitive element)
    if not check_primitive_element(p, s):
        raise ValueError(f'{s} is not a primitive element!')

    alpha = (s ** a) % p
    beta = (s ** b) % p
    key_A = (beta ** a) % p
    key_B = (alpha ** b) % p

    if steps:
        display(Math('\mathrm{Input:}'))
        display(Math(f'p = {p}, s = {s}, a = {a}, b = {b}'))
        print()
        display(Math('\mathrm{Alpha:}'))
        display(Math('\\alpha = s^{a} \;\bmod\; p'))
        display(Math(f'{alpha} = {s}^{{{a}}} \;\bmod\; {p}'))
        print()
        display(Math('\mathrm{Beta:}'))
        display(Math('\\beta = s^{b} \;\bmod\; p'))
        display(Math(f'{beta} = {s}^{{{b}}} \;\bmod\; {p}'))
        print()
        display(Math('\mathrm{Key:}'))
        display(Math('\mathcal{K} = \\alpha^{b} = \\beta^{a} = (s^{a})^{b}'))
        display(Math(f'{key_A} = {alpha}^{{{b}}} = {beta}^{{{a}}} = ({s}^{{{a}}})^{{{b}}}'))

    return (alpha, beta, key_A)


@strict_types
def caesar_chiffre(key: int, txt: str, decrypt_flag: bool = False, show_details: bool = False) -> str:
    """ Caesar chiffre. Der gegebene Text wird mithilfe einer Zahl verschlüsselt.
        Dazu wird jeder Buchstabe im Text ersetzt, mit dem Buchstaben der um die
        gegebene Zahl nach hinten versetzt im Alphabet steht. (z.B: A mit Schlüssel 1 wird B)

    Args:
      key: Schlüssel
      txt: Die zu verschlüsselnde Nachricht
      decrypt_flag: Wenn True dann wird entschluesselt, sonst wird verschluesselt
      show_details: Wenn True wird die Umwandlungstabelle angezeigt.

    Returns:
      Die ver- oder entschluesselte Version des Inputtext, abhaengig vor der decrypt_flag
    """

    mod = -1 if decrypt_flag else 1
    txt = ''.join([i for i in txt.upper() if ord(i) in range(65, 91)])

    out = ''
    for letter in txt:
        out += chr(((ord(letter) - 65 + key * mod) % 26) + 65)

    if show_details:
        alphabet = ''.join([chr(i) for i in range(65, 91)])
        key_alphabet = ''.join([alphabet[(c + key) % 26] for c in range(26)])

        print(f"{'in:':<7}{alphabet}")
        print(f"{'out:':<7}{key_alphabet}")

    return out


@strict_types
def key_word_chiffre(key_word: str, key_chr: str, txt: str, decrypt_flag: bool = False,
                     show_details: bool = False) -> str:
    """ Schluesselwortchiffre. Erstellt mithilfe eines Schluesselworts und eines
        Schluesselbuchstabens ein "neues Alphabet" mit dem eine Nachricht codiert wird.
        Das Schluesselwort wird an der Position des Schluesselbuchstabens eingefuegt,
        die im Schluesselwort vorkommenden Buchstaben werden aus dem Alphabet entfernt.
        Der Rest des Alphabets wird schliesslich hinter und vor dem Schluesselwort aufgefuellt.

    Args:
      key_word: Schluesselwort
      key_chr: Schluesselbuchstabe
      txt: Zu transformierender Text
      decrypt_flag: Wenn True dann wird entschluesselt, sonst wird verschluesselt
      show_details: Wenn True wird das "neue Alphabet" ausgegeben

    Raises:
      TypeError: Falls eines der Arguments kein string ist
      ValueError: Falls das Schluesselwort kein isogram ist und falls der Schlüsselbuchstabenstring
                  mehr als ein Buchstabe erhält.

    Returns:
      Die ver- oder entschluesselte Version des Inputtext, abhaengig vor der decrypt_flag
    """

    # Check if input is valid
    for letter in key_word:
        if key_word.count(letter) != 1:
            raise ValueError(
                f'The key_word has to be a isogram (each letter appears only once)! The letter "{letter}" has multiple appearances in "{key_word}".')

    if len(key_chr) != 1:
        raise ValueError(
            f'The key_chr has to be be a string of lenght 1. Your input was "{key_chr}" with length "{len(key_chr)}"!')

    # Clean up the input
    key_word = key_word.upper()
    key_chr = key_chr.upper()
    txt = ''.join([i for i in txt.upper() if ord(i) in range(65, 91)])

    # Get key alphabet
    alphabet = ''.join([chr(i) for i in range(65, 91)])
    remain_chr = ''.join([i for i in alphabet if i not in key_word])
    key_chr_pos = alphabet.find(key_chr)
    break_pos = len(alphabet) - len(key_word) - key_chr_pos
    key_alphabet = ''.join(remain_chr[break_pos:] + key_word + remain_chr[0:break_pos])

    # Transform input
    out = ''
    for letter in txt:
        if decrypt_flag:
            out += key_alphabet[alphabet.find(letter)]
        else:
            out += alphabet[key_alphabet.find(letter)]

    if show_details:
        print(f"{'in:':<7}{alphabet}")
        print(f"{'out:':<7}{key_alphabet}")

    return out


@strict_types
def vigenere_chiffre(key_word: str, txt: str, decrypt_flag: bool = False, show_details=False) -> str:
    """ Vignere chiffre. Der gegebene Text wird mithilfe eines Wortes verschlüsselt.
        Dazu wird das Schlüsselwort mehrfach unter den Text geschrieben. Jeder Buch-
        stabe im Text wird nun um die Position des Buchstabens im Schlüsselwort
        verschoben. (A-Z = 0-25)

    Args:
      key_word: Das Schlüsselwort
      txt: Zu transformierender Text
      decrypt_flag: Wenn True dann wird entschluesselt, sonst wird verschluesselt
      show_details: Wenn True wird die Umwandlungstabelle ausgegeben.

    Raises:
      ValueError: Falls das Schluesselwort kein Isogram ist.

    Returns:
      Die ver- oder entschluesselte Version des Inputtext, abhaengig vor der decrypt_flag
    """

    # Check if key is valid
    for letter in key_word:
        if key_word.count(letter) != 1:
            raise ValueError(
                f'The key_word has to be a isogram (each letter appears only once)! The letter "{letter}" has multiple appearances in "{key_word}".')

    mod = -1 if decrypt_flag else 1
    txt = ''.join([i for i in txt.upper() if ord(i) in range(65, 91)])
    key_word = key_word.upper()
    key_str = ''.join([key_word[nr % len(key_word)] for nr, letter in enumerate(txt)])

    out = ''
    for nr, letter in enumerate(txt):
        offset = ord(key_str[nr]) - 65
        n = (ord(letter) - 65 + offset * mod) % 26
        out += chr(n + 65)

    if show_details:
        print(f"{'in:':<7}{txt}")
        print(f"{'key:':<7}{key_str}")

    return out

