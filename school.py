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

def euler_qr_nr_criterion(p: int, a: int, steps=False) -> bool:
    """ Implementierung des Euler Kriteriums bezüglich Quadratischem Restwert und Quadratischem nicht Restwert

        Wikipedia Link: https://en.wikipedia.org/wiki/Euler%27s_criterion
    
    Args:
      p: Primzahl
      a: Teilerfremde Zahl zum Parameter "p"
      steps: Wenn True, zeigt Schritte auf
      
    Returns:
      True, wenn a einen Quadratischen Rest mod p aufweist - sonst False
    """
    if euclid_ggt(p, a) != 1:
        raise ValueError("a must be coprime to p")

    res = a**((p-1) // 2) % p

    if steps:
        display(Math(f"a^{{\\frac{{p-1}}{{2}}}} = {a}^{{\\frac{{{p}-1}}{{2}}}} = {a}^{{{(p-1) // 2}}}"))
        display(Math(f"{a}^{{\\frac{{{p}-1}}{{2}}}} \equiv {res if res == 1 else res - p} \mod {p}"))

    return True if res == 1 else False

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
      n^p % m"""
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
    return new % m

def disk_exp_func(a, b, m, log=False, steps=False):
    """Implementierung von der diskreten exponential Funktion
    
    Args(wenn log = False):
      Gleichung dieser Art von links nach rechts: 7 = 3^k mod 17
      a: linke Seite
      b: rechte seite
      m: modulo
      steps: wenn True zeigt es Schritte an

    Args(wenn log = True):
      Gleichung dieser Art von links nach rechts: k = log3(7) mod 17
      a: Basis des logs
      b: Argument im Log
      m: modulo
      steps: wenn True zeigt es Schritte an
    Returns:
      gibt k zurück für:
      Lösung für Gleichung k = loga(b) mod m 
      oder
      Lösung für Gleichung a = b^k mod m
      
      wenn keine Lösung existiert, wird None zurückgegeben"""
    if log:
        display(Math(f'"k\ =\ log_{{{a}}}({b})\ mod\ {m}"\ wird\ umgeformt\ in\ "{b}\ =\ {a}^k\ mod\ {m}"'))
        print()
        a, b = b, a
        
    tl = [[],[]]
    count=0
    for i in range(1, m+1):
        count += 1
        tl[0].append(i)
        tl[1].append((b**i)%m)
        if b**i%m == a:
            break
    if steps:
        display(Math(f"Frage\ ist:\ wann\ ist\ {a}\ =\ {b}^k\ mod\ 17"))
        print(DataFrame((tl), index=["k",f"{b}^k mod {m}"], columns=[str(" ") for x in range(count)]))
        print()
        if count == m:
            display(Math("Antwort:\ es\ gibt\ keine\ Lösung"))
        else:
            display(Math(f"Antwort:\ {a}\ =\ {b}^{count}\ mod\ {m}"))
            display(Math(f"oder\ kurz:\ k\ =\ {count}"))
    if count == m:
        return None
    else:
        return count

def qr_and_nr(n, eulersteps=False, steps=False):
    """Implementierung von quadratischem Rest und Nicht-rest
    
    Args:
      n: die Modulo Zahl für welche alles berechnet wird
      eulersteps: zeigt die schritte für die euler_phi_set an
      steps: wenn True zeigt es Schritte an
    Returns:
      quadratischer Rest und quadratischer nicht-rest"""
    numbers = sorted(euler_phi_set(n, steps=eulersteps))
    tl = [[],[]]
    for i in numbers:
        tl[0].append(i)
        tl[1].append((i**2) % n)
    if steps:
        print(DataFrame((tl), index=["x",f"x^2 mod {n}"], columns=[str(" ") for x in range(len(numbers))]))

    tl2 = [[],[]]
    qr = []
    nr = []
    for i in numbers:
        if i in tl[1]:
            a = i
            indices = [i for i, x in enumerate(tl[1]) if x == a]
            indices_new = []
            for j in indices:
                indices_new.append(tl[0][j]) 
            tl2[0].append(i)
            tl2[1].append(indices_new)
            qr.append(i)
        else:
            tl2[0].append(i)
            tl2[1].append("-")
            nr.append(i)
    if steps:
        print(DataFrame((tl2), index=["a",f"sqrt(a) mod {n}"], columns=[str(" ") for x in range(len(numbers))]))
        print()
        display(Math(f"quadratische\ rest\ QR\ = \{{{str(qr)[1:-1]} \}}"))
        display(Math(f"quadratische\ nichtrest\ NR = \{{{str(nr)[1:-1]} \}}"))

    return qr, nr