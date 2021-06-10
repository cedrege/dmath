# Documentationstyleguide: https://google.github.io/styleguide/pyguide.html
from pandas import DataFrame
from numpy import cumsum, e, ndarray, array as np_array_const, reciprocal, identity as identity_matrix
from numpy.linalg import matrix_rank, inv as matrix_inverse, det as matrix_det
from IPython.display import display, Math
from functools import reduce
from math import log, comb, factorial
from typing import get_type_hints
from sympy import Symbol, Poly, solve, Eq

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
    return factorial(n)


def binomial_coefficient(n: int, k: int, steps: bool = False) -> int:
    """ Berechnet den Binomialkoeffizienten zweier Zahlen.
        Die Paremeter wurden so implementiert, dass man n tief k lesen wuerde.
        Somit ergibt sich die Aussage: Man kann k verschiedene Objekte von der Menge n auswaehlen kann.

    Args:
      n: Anzahl Moeglichkeiten
      k: Anzahl Erfolge
      steps:          Zeigt die Schritte auf

    Raises:
      ValueError: Wenn dividend null ist

    Returns:
      Binomialkoeffizient, berechnet aus den angegeben Parametern
    """
    if steps:
        display(Math(f"\\biggl( \\begin{{matrix}} {n} \\\\ {k} \\end{{matrix}} \\biggl) = \\frac{{{n}!}}{{{k}!\cdot({n-k})!}} = {comb(n,k)}"))

    return comb(n, k)
    #dividend = 1
    #divisor = 1

    #for i in range(n, n-k, -1):
    #    dividend *= i
    #    divisor  *= ((n - i) + 1)
    
    #if divisor == 0:
    #    raise ValueError("Divisor is zero! Check parameters")

    #return dividend / divisor


def binomial_distribution(n: int, k: int, p: float, steps: bool = False) -> float:
    """ Berechnet die Binomialverteilung anhand der angegeben Werte.

    Args: 
      n: Anzahl Moeglichkeiten
      k: Anzahl Erfolge
      p: Wahrscheinlichkeit auf Erfolg bei einem Versuch
      steps:          Zeigt die Schritte auf

    Returns:
      Binomialverteilung, berechnet aus den angegeben Parametern
    """
    if steps:
        display(Math(f"\\biggl( \\begin{{matrix}} n \\\\ k \\end{{matrix}} \\biggl) \cdot p^k \cdot (1-p)^{{n-k}}"))
        display(Math(f"\\biggl( \\begin{{matrix}} {n} \\\\ {k} \\end{{matrix}} \\biggl) \cdot {p}^{{{k}}} \cdot {1-p}^{{{n-k}}} = {binomial_coefficient(n, k) * p**k * (1-p)**(n-k)}"))

    return binomial_coefficient(n, k) * p**k * (1-p)**(n-k)


def cumsum_binomial_distribution(n: int, p: float, sum_range: tuple, steps: bool = False) -> float:
    """ Berechnet die Binomialverteilung anhand der angegeben Werte.
        Die Wahrscheinlichkeiten werden in einem bestimmten Bereich gerechnet und zusammensummiert.

    Args:
      n: Anzahl Moeglichkeiten
      p: Wahrscheinlichkeit auf Erfolg bei einem Versuch
      sum_range: Bereich, indem die Wahrscheinlichkeiten zusammengerechnet werden.
                 Es wird automatisch +1 fuer die obere Grenze hinzugefuegt.
                 Manuelles hinzufügen wird in einem falschen Resultat resultieren.
      steps:          Zeigt die Schritte auf

    Returns:
      Summierte Resultate der Binomialverteilung in einem bestimmten Bereich
    """
    s = 0
    for i in range(sum_range[0], sum_range[1] + 1):
        s += binomial_distribution(n, i, p)
    
    if steps:
        display(Math(f"\sum\limits_{{i = {sum_range[0]}}}^{{{sum_range[1]}}} \\biggl( \\begin{{matrix}} {n} \\\\ i \\end{{matrix}}  \\biggl) \cdot {p}^{{i}} \cdot {1-p}^{{{n}-i}} = {s}"))

    return s


def hypergeometric_distribution(N:int, M: int, n: int, k: int, steps: bool = False) -> int:
    """ Berechnet die Hypergeometrische Verteilung anhand der angegeben Werte.

    Args:
      N: Anzahl aller Objekte
      M: Anzahl korrekter, moeglicher Ziehungen
      n: Anzahl Ziehungen
      k: Anzahl erwarteter Erfolge
      steps:          Zeigt die Schritte auf

    Returns:
      Hypergeometrische Verteilung, berechnet aus den angegeben Parametern
    """
    if steps:
        display(Math(f"P(k) = \\frac{{\\biggl( \\begin{{matrix}} M \\\\ k \\end{{matrix}} \\biggl) \cdot \\biggl( \\begin{{matrix}} N-M \\\\ n-k \\end{{matrix}} \\biggl)}}{{\\biggl( \\begin{{matrix}} N \\\\ n \\end{{matrix}} \\biggl)}}"))
        display(Math(f"P(k) = \\frac{{\\biggl( \\begin{{matrix}} {M} \\\\ {k} \\end{{matrix}} \\biggl) \cdot \\biggl( \\begin{{matrix}} {N-M} \\\\ {n-k} \\end{{matrix}} \\biggl)}}{{\\biggl( \\begin{{matrix}} {N} \\\\ {n} \\end{{matrix}} \\biggl)}}"))

    return binomial_coefficient(M, k) * binomial_coefficient(N - M, n - k) / binomial_coefficient(N, n)


def poisson_distribution(mu: float, k: int, steps: bool = False) -> float:
    """ Berechnet die Poisson Verteilung fuer my und k
    
    Args:
      mu: Anzahl Moeglichkeiten mal potenzielle Chance (n * p)
      k: Anzahl Erfolge
      steps:          Zeigt die Schritte auf

    Returns:
      Resultat der Poisson Verteilung fuer die angegeben Parameter
    """
    if steps:
        display(Math("\\frac{\mu^k}{k!} \cdot e^{(-\mu)}"))
        display(Math(f"\\frac {{{mu}^{{{k}}}}}{{{k}!}} \cdot e^{{{(-mu)}}} = {mu**k/fac(k) * e**(-mu)}"))

    return mu**k/fac(k) * e**(-mu)


def cumsum_poisson_distribution(mu: float, sum_range: tuple, steps: bool = False) -> float:
    """ Berechnet die Poisson Verteilung fuer my und k.
        Die Wahrscheinlichkeiten werden in einem bestimmten Bereich gerechnet und zusammensummiert.
    
    Args:
      mu: Anzahl Moeglichkeiten mal potenzielle Chance (n * p)
      sum_range: Bereich, indem die Wahrscheinlichkeiten zusammengerechnet werden.
                 Es wird automatisch +1 fuer die obere Grenze hinzugefuegt.
                 Manuelles hinzufügen wird in einem falschen Resultat resultieren.
      steps:          Zeigt die Schritte auf

    Returns:
      Summierte Resultate der Poisson Verteilung ueber den angegeben Bereich
    """
    s = 0
    for i in range(sum_range[0], sum_range[1] + 1):
        s += poisson_distribution(mu, i)
    
    if steps:
        display(Math(f"\sum\limits_{{i = {sum_range[0]}}}^{{{sum_range[1]}}} \\frac{{{mu}^i}}{{i!}} \cdot e^{{(-{mu})}} = {s}"))

    return s


def generating_function(result, anz_vars, limits: list = None):
    """ Generiert ein Polynom welches für diverse counting Probleme genutzt werden kann.

    Args:
         result:   Resultat des Ausdrucks. z.B (rechte Seite vom "="): x1 + x2 + x3 = 17
         anz_vars: Anzahl verschiedener Variablem im Ausdruck. z.B (linke Seite vom "="): x1 + x2 + x3 = 17
         limits:   (Optional) Upper und lower bound für die Variablen. Dies wird jeweils in einem Tuple in einer Liste angegeben. 
                   Die Einträge werden nacheinander auf die einzelnen Variablen angewandt. Beachte: wenn z.B erst ab der
                   zweiten Variable eine Einschränkung gilt, so kann diese auch ganz an den Schluss des Ausdrucks. Somit
                   spielt die Reihenfolge der Tupeln in der Liste eingetlich keine Rolle.

                   bsp:
                   1 < x1 <= 6,
                   x2,
                   4 <= x3 < 9

                   Kann so angegeben werden: [(2,6), (4,8)]
    
    Returns:
         sympy.Poly(), auf welches mit var_name.nth(11) auf die Anzahl Möglichkeiten fuer x^11 zugegriffen werden kann.
         (oder einfach das Ganze Poly anschauen)
    """
    x = Symbol("x")
    P_curr = 0
    P_arr_tmp = []

    if limits:
        for domain in limits:
            for i in range(domain[0], domain[1] + 1):
                P_curr += (x**i)
            P_arr_tmp.append(P_curr)
            P_curr = 0

        if len(limits) < anz_vars:
            for i in range(result + 1): P_curr += (x**i)
            for i in range(anz_vars - len(limits)):  P_arr_tmp.append(P_curr)
            P_curr = 0
    else:
        for i in range(result + 1):
            P_curr += (x**i)
        for j in range(anz_vars): P_arr_tmp.append(P_curr)
        P_curr = 0

    P_arr = [Poly(p) for p in P_arr_tmp]

    return reduce(lambda x,y: x * y, P_arr)


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


def erwartungswert(rmin, rmax, chance_success , intervall=1, steps = False):
    """ Berechnet den Erwartungswert

    Args:
        rmin:           Startwert des Intervalls
        rmax:           Endwert des Intervalls
        chance_success: Chance auf Erfolg bei den Versuchen
        intervall:      (Optional) Schritte in welchem sich rmin, rmax nähert. (Standard 1)
        steps:          Zeigt die Schritte auf

    Returns:
        Dict mit Erwartungswert und dessen Varianz
    """
    erg = (1 - chance_success, chance_success)

    tmp_a = []
    for i in range(rmin, rmax + 1, intervall):
        val = comb(rmax, i) * erg[0]**(rmax / intervall - i) * erg[1]**(i / intervall)
        tmp_a.append(val)
    
    if steps:
        print("VERTEILUNGSFUNKTION:")
        display(Math(f"Formel: \ \sum\limits_{{i = {rmin}}}^{{{rmax}}} \\biggl( \\begin{{matrix}} {rmax} \\\\ i \\end{{matrix}}  \\biggl) \cdot {erg[0]}^{{{rmax} - i}} \cdot {erg[1]}^{{i}} "))
        tl = [[str(x) for x in range(rmin, rmax+1)], tmp_a, cumsum(tmp_a)]
        print(DataFrame(tl, index=["Xi", "P(x = Xi)", "sum(P(x))"], columns=[str(" ") for x in range(rmin, rmax+1)]))
        print()
        print()


    tmp = []
    s = 0
    for i in range(rmin, rmax + 1, intervall):
        val = i * (comb(rmax, i) * erg[0]**(rmax / intervall - i) * erg[1]**(i / intervall))
        tmp.append(val)
        s += val
    
    if steps:
        print("ERWARTUNGSWERT:")
        display(Math(f"Formel: \ s =  \sum\limits_{{i = {rmin}}}^{{{rmax}}} i \cdot \\biggl( \\begin{{matrix}} {rmax} \\\\ i \\end{{matrix}}  \\biggl) \cdot {erg[0]}^{{{rmax} - i}} \cdot {erg[1]}^{{i}} "))
        tl = [[str(x) for x in range(rmin, rmax+1)], tmp, cumsum(tmp)]
        print(DataFrame(tl, index=["Xi", "P(x = Xi)", "sum(P(x))"], columns=[str(" ") for x in range(rmin, rmax+1)]))
        print()
        print()

    tmp_var = []
    s_var = 0
    for i in range(rmin, rmax + 1, intervall):
        val = (i-s)**2 * (comb(rmax, i) * erg[0]**(rmax / intervall - i) * erg[1]**(i / intervall))
        tmp_var.append(val)
        s_var += val
    
    if steps:
        print("VARIANZ:")
        display(Math(f"Formel: \ v = \sum\limits_{{i = {rmin}}}^{{{rmax}}} (i - s)^2 \cdot \\biggl( \\begin{{matrix}} {rmax} \\\\ i \\end{{matrix}}  \\biggl) \cdot {erg[0]}^{{{rmax} - i}} \cdot {erg[1]}^{{i}} "))
        tl = [[str(x) for x in range(rmin, rmax+1)], tmp_var, cumsum(tmp_var)]
        print(DataFrame(tl, index=["Xi", "P(x = Xi)", "sum(P(x))"], columns=[str(" ") for x in range(rmin, rmax+1)]))
        print()
        print()
    
    return {"Erwartungswert" : round(s, 13), "Varianz" : round(s_var, 13)}


def erwartungswert_infinity(zaehler: int, nenner: int, steps= False) -> float:
    """ Erwartungswert bei einer unendlichen Summe.

    Args:
      zaehler: Der Zaehler der W-keit (wenn als Bruch dargestellt)
      nenner:  Der Nenner der W-keit (wenn als Bruch dargestellt)
      steps:   Wenn True zeigt Steps an

    Returns:
      Erwartungswert bei einer unendlichen Summe.
    """
    p = zaehler / nenner
    zaehler_x = nenner - zaehler
    x = 1 - p

    if steps:
        display(Math(f"E(X) = \sum\limits_{{k=1}}^{{\infty}} k \cdot p \cdot (1-p)^{{(k-1)}} = p \cdot \sum\limits_{{k=0}}^{{\infty}} (k+1) \cdot (1-p)^k = p \cdot \\frac{{1}}{{(1 - (1-p))^2}}"))
        display(Math(f"E(X) = \sum\limits_{{k=1}}^{{\infty}} k \cdot \\biggl( \\frac{{{zaehler}}}{{{nenner}}} \\biggl) \cdot \\biggl( \\frac{{{zaehler_x}}}{{{nenner}}} \\biggl)^{{(k-1)}} = \\frac{{{zaehler}}}{{{nenner}}} \cdot \sum\limits_{{k=0}}^{{\infty}} (k+1) \cdot \\biggl( \\frac{{{zaehler_x}}}{{{nenner}}} \\biggl)^k = \\frac{{{zaehler}}}{{{nenner}}} \cdot \\frac{{1}}{{(1 - \\frac{{{zaehler_x}}}{{{nenner}}})^2}} = {round(p * (1 / (1 - x)**2), 13)}"))

    return round(p * (1 / (1 - x)**2), 13)


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


def prime_fac_eu_phi(n, steps=False):
    """ Berechnet die Primfaktorzerlegung anhad der eulerischen Phi Funktion

    Args:
       n:     Produkt von zwei Primzahlen p und q
       steps: Zeigt Schritte auf
    """
    eu_phi = euler_phi(n)
    q = Symbol("q")
    _e = Eq(n - ((n/q)+q)+1, eu_phi)

    solved = solve(_e)

    if steps:
        display(Math(f"{eu_phi} = {n} - \\biggl( \\frac{{{n}}}{{q}} + q \\biggl) +1 \ \\Rightarrow \ q^2 +{eu_phi - n if eu_phi - n >= 0 else f'({eu_phi - n})'} \cdot q + {n} = 0"))
        display(Math("Solve \ for \ q"))
        display(Math(f"q = {solved[0]}, \ p = {solved[1]}"))

    return solved


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
    first_round = a // b
    return euclid_ggt_print(a, b, steps, first_round)


def euclid_ggt_buergi(a,b, steps = False):
    a_input = a
    b_input = b
    u_old, u = 1, 0; v_old, v = 0, 1
    if steps:
        display(Math(f"{a}\ |\ {'-'}\ |\  {u_old}\ |\ {v_old}"))
    while b:
        q = a // b
        u, u_old = u_old - q * u, u
        v, v_old = v_old - q * v, v
        a, b = b, a % b
        if steps:
            display(Math(f"{a}\ |\ {q}\ |\  {u_old}\ |\ {v_old}"))
    if steps:
        display(Math(f"{u_old*a_input + v_old * b_input} = {u_old} * {a_input} + {v_old} * {b_input}"))
    d, s, t = euclid_ggt_extended_print(a_input, b_input)
    s_1 = s
    t_1 = t
    if s < 0: s_1 = s + b_input
    if t < 0: t_1 = t + a_input

    print()
    if steps:
        display(Math(f"{a_input}^{{-1}}\mod {b_input} = {s_1}"))
        display(Math(f"{b_input}^{{-1}}\mod {a_input} = {t_1}"))
    return a, s_1, t_1


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
        if steps:
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


def small_fermat(a: int,b: int,x: int, steps=False):
    """implementierung für step by step Anleitung des kleinen Fermats
    
    Args:
      a: Basis
      b: Exponent
      x: zu welchem modulo
      steps: Wenn True, zeigt Schritte auf
      
    Returns:
      a^b mod x wenn alg durchführbar
      sonst: False
      
    """
    if check_prime(x):
        new_exp = b // (x-1)
        exp_rest = b % (x-1)
        display(Math(f'{a}^{{{b}}}\ mod\ {x}'))
        if steps and exp_rest > 0:
            display(Math(f'(\f{a}^{{{x-1}}})^{{{new_exp}}}\ *\ {a}^{{{exp_rest}}} \ mod\ {x}'))
            display(Math(f'(\f{a}^{{{x-1}}}\ mod\ {x})^{{{new_exp}}}\ *\ {a}^{{{exp_rest}}}\ mod\ {x}'))
            display(Math('Kleiner\ Satz\ von\ Fermat\ =>\ m^{(p-1)}\ mod\ p\ =\ 1\ ||\ wenn\ p\ =\ prime'))
            display(Math(f'1^{{{new_exp}}}*\ {a}^{{{exp_rest}}}\ mod\ {x}'))
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

        return a**b % x
        
    else:
        print('Kleiner Fermat funktioniert nur wenn mod einer Primzahl')
        display(Math(f'{a}^{{{b}}}\ mod\ {x}'))
        display(Math(f'{a**b%x}\ mod\ {x}'))
        
        return False


def euler_prime(x: int, y=0, steps=False):
    """Berechnung, wieviele Primzahlen in einem Bereich anzutreffen sind
    
    Args:
      x: Ende des Intervalls
      y: Anfang des intervalls, von 0 her falls nicht angegeben
    Returns:
      erwartete Primzahlen in bereich zwischen x und y
      
    """
    if steps:
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
    return x/(log(x)) - y/(log(y))


def bayes(fpr, fnr, verb, steps = False): #oder 1-spezifität, 1-sensitivität, verb
    """Implementierung von Bayes rule
    
    Args:
      fpr: False positive rate
      fnr: False negative rate
      verb: Verbreitung
      
    Returns:
      alle möglichen chancen nach erfolgtem Test"""
    display(Math("Bayes\ Rule:\ P(A|B)\ =\ \\frac{P(B|A)*P(A)}{P(B)}"))
    sensitivity = 1-fnr
    spezifitaet = 1-fpr
    if steps:
      display(Math(f"P(+|T+)\ =>\ \\frac{{{f'{sensitivity} * {verb}'}}}{{{f'{sensitivity} * {(verb)} + {fpr} * {1-verb}'}}}\ =\ {sensitivity * verb / (sensitivity * verb + fpr * (1-verb))}"))
      display(Math(f"P(-|T+)\ =>\ 1-{sensitivity * verb / (sensitivity * verb + fpr * (1-verb))}\ =\ {1 - (sensitivity * verb / (sensitivity * verb + fpr * (1-verb)))}"))
      display(Math(f"P(-|T-)\ =>\ \\frac{{{f'{spezifitaet} * {1-verb}'}}}{{{f'{spezifitaet} * {(1-verb)} + {fnr} * {verb}'}}}\ =\ {spezifitaet * (1-verb) / (spezifitaet * (1-verb) + fnr * verb)}"))
      display(Math(f"P(+|T-)\ =>\ 1-{spezifitaet * (1-verb) / (spezifitaet * (1-verb) + fnr * verb)}\ =\ {1- (spezifitaet * (1-verb) / (spezifitaet * (1-verb) + fnr * verb))}"))

      display(Math(f"P(T+)\ =>\ {verb}*{sensitivity}+{1-verb}*{fpr}\ =\ {verb * sensitivity + (1-verb) * fpr} "))
      display(Math(f"P(T-)\ =>\ {verb}*{fnr}+{1-verb}*{spezifitaet}\ =\ {verb * fnr + (1-verb) * spezifitaet} "))


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
        display(Math(f'{new % m}\ mod\ {m}'))
    return new % m


#@strict_types
def check_primitive_element(p: int, s: int, steps=False) -> bool:
    """ Diese Funktion ueberprueft, ob der Integer s ein primitives Element
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
            display(Math(f'{sequence[-1]} \cdot {s} \;\\bmod\; {p} = {sequence[-1] * s % p}'))

        sequence.append(sequence[-1] * s % p)

    sequence = list((set(sorted(sequence))))

    if sequence == Z:
        return True
    else:
        return False


#@strict_types
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


#@strict_types
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
        display(Math('\\alpha = s^{a} \;\\bmod\; p'))
        display(Math(f'{alpha} = {s}^{{{a}}} \;\\bmod\; {p}'))
        print()
        display(Math('\mathrm{Beta:}'))
        display(Math('\\beta = s^{b} \;\\bmod\; p'))
        display(Math(f'{beta} = {s}^{{{b}}} \;\\bmod\; {p}'))
        print()
        display(Math('\mathrm{Key:}'))
        display(Math('\mathcal{K} = \\alpha^{b} = \\beta^{a} = (s^{a})^{b}'))
        display(Math(f'{key_A} = {alpha}^{{{b}}} = {beta}^{{{a}}} = ({s}^{{{a}}})^{{{b}}}'))

    return (alpha, beta, key_A)


#@strict_types
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


#@strict_types
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


#@strict_types
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


#@strict_types
def gen_prime(i:int)-> int:
    """ Primzahlengenerator. Liefert die erste Primzahl nach i oder i selber falls i
        eine Primzahl ist.
    
    Args:
      i: Integer von dem aus eine Primzahl gesucht wird.  

    Returns:
      Primzahl
    """
    found = False
    i-=1
    
    while not found:
        i+=1
        found = check_prime(i)
        
    return i


def rsa_keygen(p: int, q: int, e: bool = None, d: bool = None, steps=False)-> tuple:
    """ RSA Keygeneration. Nimmt Primzahlen p und q und Schlüssel e und d. 
        Falls e und d nicht gegeben sind werden sie generiert.
    
    Args:
        p: Primzahl
        q: Primzahl
        e: Schlüssel
        d: Schlüssel
      
    Returns:
      Das Produkt von q und p = n, sowie den privaten Schuessel d sowie den öffentlichen 
      Schluessel d.
    """
    
    n = p*q
    n_phi = (p-1)*(q-1)
    if steps:
        display(Math(f'n\ =\ {p}*{q}\ |\ =\ {p*q}'))
        display(Math(f'\Phi({str(n)})\ =\ {(p-1)}*{(q-1)}\ | =\ {(p-1)*(q-1)}'))
    
    if e is None:
        e = int(n_phi/4)
    
    if d is None:
        while euclid_ggt(n_phi, e) != 1:
            e-=1
        print()
        if e > n_phi:
            d = euclid_ggt_buergi(e, n_phi, steps)[1]
        else:
            d = euclid_ggt_buergi(n_phi, e, steps)[2]
    if steps:
        print(f"{'Primzahl p:':<15}{p:>20}")
        print(f"{'Primzahl q:':<15}{q:>20}")
        print(f"{'Produkt n:':<15}{n:>20}")
        print(f"{'e. phi von n:':<15}{n_phi:>20}")
        print()
        print(f"{'priv. key d:':<15}{d:>20}")
        print(f"{'oeff. key e:':<15}{e:>20}")
  
    return n, e, d


#@strict_types
def rsa(n: int, k: int, m: int)-> int:
    """ Diese Funktion Ver- oder Entschluesselt die Nachricht m mithilfe des Primzahlenprodukts n
        und des (oeffentlichen oder privaten) Schluessels.
    
    Args:
        n: Primzahlenprodukt aus p und q  
        k: Schluessel d oder e
        m: 'Nachricht' in Form eines Integers, welche ver- oder entschluesselt werden soll.
        
    Raises:
        ValueError: Falls n <= m ist.

    Returns:
        Ver- oder entschluesselte Nachricht.
    """
    
    if n <= m:
        raise ValueError(f'm has to be smaller than n! ({m} !< {n})')
        
    return (m**k) % n


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
        display(Math(f"Frage\ ist:\ wann\ ist\ {a}\ =\ {b}^k\ mod\ {m}"))
        print(DataFrame((tl), index=["k",f"{b}^k mod {m}"], columns=[str(" ") for x in range(count)]))
        print()
        if count == m:
            display(Math("Antwort:\ es\ gibt\ keine\ Lösung"))
        else:
            display(Math(f"Antwort:\ {a}\ =\ {b}^{{{count}}}\ mod\ {m}"))
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


def euler_theorem(a: int,b: int,x: int, steps=False, primsteps=False):
    """implementierung für step by step Anleitung des Satzes von Euler
    
    Args:
      a: Basis
      b: Exponent
      x: zu welchem modulo
      steps: Wenn True, zeigt Schritte auf
      
    Returns:
      a^b mod x wenn alg durchführbar
      sonst: False
    """
    if b >= euler_phi(x) and euclid_ggt(a,x) == 1:
        if steps:
            euler = euler_phi(x)
            new_exp = b // (euler_phi(x))
            exp_rest = b % (euler_phi(x))
            euler_phi(x, steps=steps, primfac_steps=primsteps)
            display(Math(f'{a}^{{{b}}}\ =\ {a**b%x}\ mod\ {x}'))
            display(Math(f'({a}^{{{euler}}})^{{{new_exp}}}\ *\ {a}^{{{exp_rest}}}\ mod\ {x}'))
            display(Math(f'({a}^{{{euler}}}\ mod\ {x})^{{{new_exp}}}\ *\ {a}^{{{exp_rest}}}\ mod\ {x}'))
            display(Math('Satz\ von\ Euler\ =>\ a^{\Phi(n)}\ =\ 1\ mod\ n'))
            display(Math(f'1^{{{new_exp}}}\ *\ {a}^{{{exp_rest}}}\ mod\ {x}'))
            display(Math(f'{a}^{{{exp_rest}}}\ mod\ {x}'))
            display(Math(f'{a**exp_rest}\ mod\ {x}'))

        return a**b % x

    if b <= euler_phi(x):
        raise ValueError(f"exponent muss grösser als phi({x}) == {euler_phi(x)} sein")
        

    if euclid_ggt(a,x) != 1:
        raise ValueError(f'{a} und {x} sind nicht teilerfrenmd. ggt = {euclid_ggt(a,x)} also != 1 => use square and multiply (sma)')


def page_rank(connection_matrix: ndarray, dampening_factor_zaehler: float, dampening_factor_nenner: float, steps=False) -> ndarray:
    """ Berechnet den PageRank eines Graphen anhand der Adjunkten matrix.

    Args:
        connection_matrix:        Matrix, die die Verbindungen zu den einzelnen Nodes repraesentiert
                                  Diese Matrix wird immer nach dem Schema "Column verbindet Row" aufgebau.
                                  Diese muss richtig erstellt werden sonst ist das Resultat falsch. (Nicht nur Zahlen sondern auch die Row Column)
        dampening_factor_zaehler: Der Zaehler des Daempfungsfaktors der Aufgabe. (Wenn als Bruch dargestellt)
        dampening_factor_nenner:  Der Nenner des Daempfungsfaktors der Aufgabe. (Wenn als Bruch dargestellt)
        steps:                    (Optional) gibt den Rechenweg aus.

    Returns:
        Vector mit den Page-Rank Werten.
    """

    dampening_factor = dampening_factor_zaehler / (dampening_factor_nenner if dampening_factor_nenner > 0 else 1)

    if dampening_factor > 1:
        raise ValueError("Dampening factor must be between 0 and 1.")

    if connection_matrix.shape[0] != connection_matrix.shape[1]:
        raise ValueError("Connection matrix must be a square matrix.")

    dampening_complement = 1 - dampening_factor
    connection_mx_size = int(connection_matrix.shape[0])

    # PR vector with unknown vars
    pr = [f"PR_{x+1}" for x in range(connection_mx_size)]
    
    # calculate vector b
    b = np_array_const([(1 / connection_mx_size) for x in range(connection_mx_size)]).reshape(connection_mx_size, 1)

    #dampening_complement = (dampening_factor_zaehler - dampening_factor_nenner) / dampening_factor_zaehler
    I = (identity_matrix(connection_mx_size))
    A = (connection_matrix * dampening_factor)
    A_tilde = I - A

    if matrix_rank(A_tilde) != connection_mx_size:
        raise ValueError("Rank of connection_matrix_tilde is not equal to the shape of connection_matrix. Most likely a typo in your connection_matrix.")
    
    r = matrix_inverse(A_tilde * dampening_factor_nenner) @ (b)

    if steps:
        pr_dis_str = " \\\\ ".join(pr)
        display(Math(f"\\vec{{r}} = \\begin{{bmatrix}} {pr_dis_str} \\end{{bmatrix}}"))
        b_dis_str = " \\\\ ".join([f"\\frac{{1}}{{{connection_mx_size}}}" for x in b.flat])
        display(Math(f"\\vec{{b}} = \\frac{{{dampening_factor_nenner - dampening_factor_zaehler}}}{{{dampening_factor_nenner}}} \cdot \\begin{{bmatrix}} {b_dis_str} \\end{{bmatrix}}"))

        # Print calc way:
        display(Math(f"I \cdot \\vec{{r}} = (\\frac{{{dampening_factor_zaehler}}}{{{dampening_factor_nenner}}} \cdot A) \cdot \\vec{{r}} + \\frac{{{dampening_factor_nenner - dampening_factor_zaehler}}}{{{dampening_factor_nenner}}} \cdot \\vec{{b}}"))
        display(Math(f"(I -(\\frac{{{dampening_factor_zaehler}}}{{{dampening_factor_nenner}}} \cdot A)) \cdot \\vec{{r}} = \\frac{{{dampening_factor_nenner - dampening_factor_zaehler}}}{{{dampening_factor_nenner}}} \cdot \\vec{{b}}"))
        display(Math(f"\\underbrace{{{dampening_factor_nenner}(I - (\\frac{{{dampening_factor_zaehler}}}{{{dampening_factor_nenner}}} \cdot A))}}_{{\\tilde{{A}}}} \cdot \\vec{{r}} = \\vec{{b}}"))
        display(Math(f"\\vec{{r}} = \\tilde{{A}}^{{-1}} \cdot \\vec{{b}}"))

        # Print r
        r_dis_str = " \\\\ ".join(str(x) for x in r.flat)
        display(Math(f"\\vec{{r}} = \\begin{{bmatrix}} {r_dis_str} \\end{{bmatrix}}"))

    return r


def amount_spanning_trees(A: ndarray, D: ndarray, steps: bool = False) -> int:
    """Berechnet die Anzahl Spannbäume im Graphen mit dem Linalg Weg

    Args:
        A: Adjugate matrix vom Graphen
        D: Degree matrix vom Graphen
        steps: Ausgabe des Rechenweges
    
    Returns:
        Maximale Anzahl an Spannbäumen
    """
    L = (D - A)[:-1,:-1]
    
    if steps:
        display(Math("L = D - A"))
        display(Math("Eine \ Dimension \ entfernen"))
        display(Math(f"det(L) = {int(matrix_det(L))}"))

    return int(matrix_det(L))


def display_bayes():
    display(Math("Bayes\ Rule:\ P(A|B)\ =\ \\frac{P(B|A)*P(A)}{P(B)}"))
    display(Math("Bayes\ Rule\ extended:\ P(A|B)\ =\ \\frac{P(B|A)*P(A)}{P(B|A)*P(A)+P(B|\\neg A)* P(\\neg A))}"))


def derangements(n, steps=False):
    """Implementierung von derangements
    
    Args:
      n = anzahl verschiedener Einheiten
      
    Returns:
      anzahl der möglichen Derangements"""
    text=""
    text_refined=""
    a = 0
    for i in range(n+1):
        a += (-1)**i*fac(n)/fac(i)
        if steps:
            if i != n:
                text += f"\\frac{{{f'(-1)^{{{i}}} * {n}!'}}}{{{f'{i}!'}}}\ +\ "
                
            else:
                text += f"\\frac{{{f'(-1)^{{{i}}} * {n}!'}}}{{{f'{i}!'}}}"
            if i == 0:
                text_refined += f"\\frac{{{fac(n)}}}{{{fac(i)}}}\ "
            else:
                if i%2 == 0:
                    text_refined += f"+\ \\frac{{{fac(n)}}}{{{fac(i)}}}\ "
                else:
                    text_refined += f"-\ \\frac{{{fac(n)}}}{{{fac(i)}}}\ "
    if steps:
        display(Math(text))
        display(Math(text_refined))
    return a


def nullteiler(n):
    """Nullteiler
    
    Args:
      n = Nummber welche auf Nullteiler geprüft werden
      
    Returns:
      Set mit den Nullteilern"""

    a= []
    for i in range(1,n):
        for j in range(1,n):
            if i*j % n == 0:
                a.append(i)
                display(Math(f"{i}*{j}=0\ mod\ {n}"))
    a  = set(a)
    display(Math(f"Nullteiler\ sind:\ {a}"))
    return set(a)

def uniq(input):
  output = []
  for x in input:
    if x not in output:
      output.append(x)
  return output


def perf_savety(function: list, chance_messages: dict, chance_keys: dict, latex_display=True):
    """implementierung für step by step Anleitung der Perfekten Sicherheit
    
    Args:
      function: Eine Liste die folgendermassen gefüllt wird: Key, message, cyper(welche bei der Verschlüsselung mit dem key und der Nachricht entsteht)
        --> bsp: mit wenn die Nachricht a mit dem key1 mit B verschlüsselt wird, dann wird eingegeben ["k1", "a", "B"]
            also in worten "k1 macht aus der Nachricht a den Cypher B" 
      chance_messages: ein Dictionary mit den Nachrichten in Verbindung mit der Wahrscheinlichkeit, dass diese auftritt
      chance_keys: ein Dictionary mit den Keys in Verbindung mit der Wahrscheinlichkeit, dass diese auftritt
      latex_display: Wenn True, zeigt Schritte in latex an (führt zu schlechterer Perfomance)
      
    Returns:
      Alle Kalkulationen für die Perfekte Sicherheit"""

    function_len = len(chance_messages)*len(chance_keys)
    messages = []
    cypher_text = []
    keys = []
    for i in range(function_len):
        if latex_display:
            display(Math(f'function({function[i*3]},\ {function[i*3+1]})\ =\ {function[i*3+2]}'))
        else:
            print(f"function ({function[i*3]}, {function[i*3+1]}) = {function[i*3+2]}")
        keys.append(function[i*3])
        messages.append(function[i*3+1])
        cypher_text.append(function[i*3+2])
    cypher_text_uniq = uniq(cypher_text)
    keys_uniq = uniq(keys)
    messages_uniqu = uniq(messages)

    indices_cyper_text = {}
    for j in cypher_text_uniq:
        indices = [i for i, x in enumerate(function) if x == j]
        indices_cyper_text[f"{j}"]= indices
    chances_cypher = {}

    for i in cypher_text_uniq:
        chance_for_cypher_text = 0
        for j in indices_cyper_text[i]:
            if latex_display:
                display(Math(f"p({i} )\ ==\ {function[j-1]}"))
            else:
                print(f"p({i}) == {function[j-1]} ")
            current_message = function[j-1]
            current_key = function[j-2]
            current_key_chance = chance_keys[current_key]
            if latex_display:
                display(Math(f'\\frac{{1}}{{{int(1/current_key_chance)}}}\ *\ \\frac{{1}}{{{int(1/chance_messages[current_message])}}}\ =\ \\frac{{1}}{{{int(1/(current_key_chance * chance_messages[current_message]))}}}'))
            else:
                print(f"{current_key_chance} * {chance_messages[current_message]}")
            chance_for_cypher_text += chance_messages[current_message]*current_key_chance
        if latex_display:
            display(Math(f"chance\ for\ p({i})\ =\ {round(chance_for_cypher_text,15)}"))
        else:
            print(f"chance for p({i}) = {round(chance_for_cypher_text,15)}")
        print()
        chances_cypher[i]=(round(chance_for_cypher_text, 15))

    indices_messages = {}
    for j in chance_messages:
        indices = [i for i, x in enumerate(function) if x == j]
        indices_messages[j]= indices
 
    display(Math("Bayes\ Rule:\ P(A|B)\ =\ \\frac{P(B|A)*P(A)}{P(B)}"))
    display(Math("bei\ mehr\ als\ einer\ Chance,\ müssen\ sie\ addiert\ werden,\ bei\ keiner\ ist\ die\ Chance\ 0"))
    for current_message in indices_messages:
        for current_cypher in cypher_text_uniq:
            if latex_display:
                display(Math(f"Cyper:{current_cypher}\ Nachricht:{current_message}"))
            else:
                print("current cypher", current_cypher)
                print("current message", current_message)
            for i in range(len(chance_keys)):
                if current_cypher == function[indices_messages[current_message][i]+1]:
                    if latex_display:
                        display(Math(f"{chance_messages[current_message]} * {chance_keys[function[indices_messages[current_message][i]-1]]}/{chances_cypher[current_cypher]}"))
                        display(Math(f"Chance\ for\ ( {current_message} | {current_cypher} )\ =\ {chance_messages[current_message] * chance_keys[function[indices_messages[current_message][i]-1]] / chances_cypher[current_cypher]}"))
                    else:
                        print(f"chance for ( {current_message} | {current_cypher} ) = {chance_messages[current_message] * chance_keys[function[indices_messages[current_message][i]-1]] / chances_cypher[current_cypher]}")
            print()
                    
                    
