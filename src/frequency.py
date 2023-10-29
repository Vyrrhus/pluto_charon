""" Base class for Combinations & Frequencies
"""
from __future__ import annotations
from typing import List, Tuple, Dict, TYPE_CHECKING
import itertools as it
import re
import numpy as np
import pandas as pd
from inspect import cleandoc
if TYPE_CHECKING:
    from .simulation import Simulation
    from .system import System

########################################################################
#                       COMBINATIONS
########################################################################
class CoeffGenerator:
    """ Generator class of sets of integers. """
    def __init__(self,
                 nb_elements: int,
                 nb_non_zero: int,
                 kmax: int,
                 withGCD: bool = True,
                 filterWith: List[List[int]] = [],
                 isSymetric: bool = False
                 ) -> None:
        """ Constructor of the generator.

            ### Parameters
            - `nb_elements`: number of elements in the set.
            - `nb_non_zero`: number of non zero terms within the set.
            - `kmax`: non zero terms ∈ [-kmax ; kmax]
            - `with_GCD`: if True, sets all have a GCD ≤ 1
            - `filterWith`: list of elements that must be non zero.
            - `isSymetric`: first non zero element must be positive.
        """
        # Parameters of the set
        self.nb_elements = nb_elements
        self.nb_non_zero = nb_non_zero
        self.kmax        = kmax
        self.withGCD     = withGCD
        self.isSymetric  = isSymetric

        self.coeff_range = [(ii // 2 if ii % 2 == 0 else -ii // 2) 
                            for ii in range(0, 2 * kmax + 1)]
        self.non_zero_combinations = it.combinations(range(nb_elements),
                                                     nb_non_zero)
        
        # Filters
        if filterWith:
            self.non_zero_combinations = (
                comb for comb in self.non_zero_combinations
                if all(any(value in comb for value in valuesToFilter)
                       for valuesToFilter in filterWith)
            )

        # Initialize generator
        self.non_zero_values: List[Tuple[int, ...]] = []
        self.combinations_iter = None

    def __link(self,
              values: Tuple[int, ...],
              non_zero_idx: Tuple[int, ...]
              ) -> Tuple[int, ...]:
        """ Associate the non zero elements and their values. """
        result =  [0] * self.nb_elements
        for idx, value in zip(non_zero_idx, values):
            result[idx] = value
        return tuple(result)

    def __iter__(self) -> CoeffGenerator:
        return self
    
    def __next__(self) -> Tuple[int, ...]:
        """ Next element of the generator. """
        while True:
            # Set all new possible permutations in an iterator
            if (self.combinations_iter is None 
                or not self.non_zero_values):
                try:
                    # Non zero index of the set:
                    non_zero_idx = next(self.non_zero_combinations)
                    
                    # First element can be either positive or negative
                    self.non_zero_values = list(
                        it.product(self.coeff_range[1:],
                                    repeat = self.nb_non_zero)
                    )
                    
                    # Ensure first element is always positive
                    if self.isSymetric:
                        self.non_zero_values = [
                            el 
                            for el in self.non_zero_values
                            if el[0] > 0]
                    
                    # Create the iterator
                    self.combinations_iter = (
                        self.__link(non_zero_value, non_zero_idx)
                        for non_zero_value in self.non_zero_values)

                except StopIteration:
                    raise StopIteration("No more non-zero permutations")
            
            # Iterate over it to find a new candidate.
            try:
                while True:
                    candidate = next(self.combinations_iter)

                    # Check GCD
                    if np.gcd.reduce(candidate) in [0, 1] or not self.withGCD:
                        return candidate

            # No more element in the iterator
            except StopIteration:
                self.combinations_iter = None
                self.non_zero_values = []

class BaseCombination():
    """ Linear combination of elements
    """
    def __init__(
            self,
            coefficients: List[int],
            objects: List[object]
        ) -> None:
        """ Base combination constructor. """
        self.coefficients = [
            coeff 
            for coeff in coefficients 
            if coeff != 0]
        
        self.elements     = [
            el 
            for coeff, el in zip(coefficients, objects) 
            if coeff != 0]
    
    @property
    def size(self) -> int:
        """ Number of elements in the combination. """
        return len(self.coefficients)

    @property
    def value(self) -> float:
        """ Algebric value of the combination. """
        return np.sum([coeff * el
                       for coeff, el
                       in zip(self.coefficients, self.elements)])
    
    def __str__(self) -> str:
        """ String representation of the combination. """
        signs = ['+' if coef > 0 else '-' for coef in self.coefficients]
        scals = list(map(abs, self.coefficients))

        # Null combination
        if not self.elements:
            return '*'
        
        repr = ''.join([(
            f"{(signs[ii] if signs[ii] == '-' or ii != 0 else ' ')}"
            f" {str(scals[ii]) + ' ' if scals[ii] > 1 else ''}"
            f"{str(self.elements[ii])} "
        ) for ii in range(self.size)])

        return repr

class FreqCombination(BaseCombination):
    """ Combination of frequencies. """
    def __init__(
            self, 
            coefficients: List[int],
            frequencies: List[FundamentalFrequency]
        ) -> None:
        """ Frequencies combination constructor. """
        super().__init__(coefficients, frequencies)
        self.elements: List[FundamentalFrequency]
    
    def angle(
            self,
            simulation: Simulation,
            inDegree: bool = False
        ) -> np.ndarray:
        """ Compute the combination's corresponding angle. """
        angle = np.zeros_like(simulation.time)

        # Compute combination
        for coeff, freq in zip(self.coefficients, self.elements):
            name = freq.body
            key  = freq.angle

            body     = simulation.system.bodies[name]
            elements = body.elements(simulation.system,
                                     simulation.simulation.G)
            
            angle += coeff * np.unwrap(elements[key])
        
        # Return angle
        if inDegree:    return np.rad2deg(angle) % (360)
        else:           return angle % (2 * np.pi)

    @property
    def LaTeX(self) -> str:
        """ LaTeX representation of the combination. """
        string = str(self)
        for expression in (
            ("v", r"\nu"),
            ("k", r"\kappa"),
            ("Om", r"\Omega"),
            ("PC", r"{bin}"),
            ("pi", r"\varpi")
            ):
            string = string.replace(*expression)
        
        return f"${string}$"

    @staticmethod
    def read(
        expression: str,
        path: str
        ) -> FreqCombination:
        """ Convert literal expression in FreqCombination object. """
        regexp = r'([-+]?\s*\d*\.?\d*)\s*([a-zA-Z_]+)'
        terms = re.findall(regexp, expression)
        fundamentals = FundamentalFrequency.load(path)
        fundamentals_dict = {freq.name: freq for freq in fundamentals}

        frequencies  = []
        coefficients = []

        for term in terms:
            coefficient = term[0].strip()
            if coefficient in ['', '+']:
                coefficient = '1'
            if coefficient == '-':
                coefficient = '-1'
            freqName = term[1].strip()

            frequencies.append(fundamentals_dict[freqName])
            coefficients.append(int(coefficient.replace(" ", "")))
        
        return FreqCombination(coefficients, frequencies)

    @staticmethod
    def find_resonance(
        fundamentals: List[FundamentalFrequency],
        kmax: int = 5,
        nbMinNonZero: int = 1,
        nbMaxNonZero: int | None = None,
        threshold: float = np.deg2rad(1)
        ):
        """ Returns a list of combinations that could be associated
            with a resonance.
        """
        # Verbosity
        print("Fundamental frequencies used : ")
        print("===============================")
        for freq in fundamentals:
            print(f"{str(freq):<8} | {freq.freq:.15f}")

        # Search candidates with a small enough value
        candidates = Frequency(0, []).search(
            fundamentals,
            kmax = kmax,
            nbMinNonZero = nbMinNonZero,
            nbMaxNonZero = nbMaxNonZero or len(fundamentals),
            withRatio = True,
            isSymetric = True,
            searchResonance = True,
            thresholdResonance = threshold
        )

        return candidates
    
        # minimumRate = abs(min([el.value for el in candidates]))
        # maximumDays = 2 * np.pi / minimumRate

        # # Simulation to find resonant angles
        # sim = Simulation(syst)
        # sim.integrate(maximumDays, 0.5)

        # from matplotlib import pyplot as plt
        # fig, ax = plt.subplots()

        # for candidate in candidates:
        #     angle = candidate.angle(sim, inDegree=True)
        #     ax.plot(sim.time, angle, label=candidate.latex)
        
        # ax.grid()
        # ax.legend()

        # plt.show()

class Combination():
    """ Linear combination of frequencies
    """
    def __init__(
            self,
            coefficients: List[int],
            fundamentals: List[FundamentalFrequency]
    ) -> None:
        """ Combination constructor. """
        # Pairs of coefficients and fundamental frequencies
        self.coefficients = coefficients
        self.fundamentals = fundamentals
        
    @property
    def value(self) -> float:
        """ Algebric value of the combination. """
        return np.sum([freq * coeff
                       for coeff, freq 
                       in zip(self.coefficients, self.fundamentals)])

    @property
    def LaTeX(self) -> str:
        """ LaTeX representation of the combination. """
        string = str(self)
        for litt in (("v", r"\nu"), 
                     ("k", r"\kappa"), 
                     ("Om", r"\Omega"), 
                     ("PC", r"_{pc}")):
            string = string.replace(*litt)

        return f"${string}$"

    def libration(
            self,
            simulation: Simulation,
            inDegree: bool = False,
            test=False
        ) -> np.ndarray:
        """ Compute the combination's corresponding angle. """
        # Initialize
        angle = np.zeros_like(simulation.time)

        # Link angles <-> frequencies
        for coeff, freq in zip(self.coefficients, self.fundamentals):
            # Binary
            if "PC" in freq.name or "bin" in freq.name:
                name = "charon"

            # Moons
            else:
                name = {"S": "styx",
                        "N": "nix",
                        "K": "kerberos",
                        "H": "hydra"}[freq.name[-1]]
            
            body = simulation.system.bodies[name]

            # True longitudes
            if freq.name[0] == "n":
                long = body.elements(
                    simulation.system, 
                    simulation.simulation.G
                    ).long
                
                angle += coeff * np.unwrap(long)
            
            elif 'pi' in freq.name:
                z = body.elements(
                    simulation.system,
                    simulation.simulation.G
                    ).z
                k, h = np.real(z), np.imag(z)
                e  = np.sqrt(k**2 + h**2)
                pi = np.arctan2(h/e, k/e)

                angle += coeff * np.unwrap(pi)
            
            elif 'om' in freq.name.lower():
                elems = body.elements(
                    simulation.system,
                    simulation.simulation.G
                )
                _, _, _, Omega, _, _ = elems.osculating

                angle += coeff * np.unwrap(Omega)

        # Finally
        if inDegree:
            return np.rad2deg(angle) % 360
        else:
            return angle % (2*np.pi)

    @staticmethod
    def read(
        expression: str,
        fundamentals_path: str
        ) -> Combination:
        """ Convert literal expression in Combination() object. """
        pattern = r'([-+]?\s*\d*\.?\d*)\s*([a-zA-Z_]+)'
        terms = re.findall(pattern, expression)
        fundamentals = FundamentalFrequency.load(fundamentals_path)
        fundamentals_dict = {freq.name: freq for freq in fundamentals}

        frequencies  = []
        coefficients = []

        for term in terms:
            coefficient = term[0].strip()
            if coefficient in ['', '+']:
                coefficient = '1'
            if coefficient == '-':
                coefficient = '-1'
            freqName = term[1].strip()

            frequencies.append(fundamentals_dict[freqName])
            coefficients.append(int(coefficient.replace(" ", "")))
        
        return Combination(coefficients, frequencies)

    def __str__(self):
        names  = [freq.name for freq in self.fundamentals]
        signs  = ['+' if value > 0 else '-' for value in self.coefficients]
        values = list(map(abs, self.coefficients))

        if not names:
            return '*'

        return ''.join([
            (f"{(signs[ii] if signs[ii] == '-' or ii != 0 else ' ')}"
             f" {str(values[ii]) + ' ' if values[ii] > 1 else ''}"
             f"{names[ii]} "
             ) for ii in range(len(names))])

########################################################################
#                       FREQUENCIES
########################################################################
class BaseFrequency():
    """ Frequency base class. """
    def __init__(self,
                 frequency: float,
                 phase: float | None = None,
                 amplitude: float | None = None) -> None:
        """ Base constructor. """
        self.freq  = frequency
        self.phase = phase
        self.amp   = amplitude

    @property
    def period(self):
        """ Period ie inverse of frequency. """
        return 2 * np.pi / self.freq if self.freq != 0 else 0

class Frequency(BaseFrequency):
    """ Frequency base class. """
    def __init__(self, 
                 frequency: float,
                 fundamentals: List[FundamentalFrequency],
                 amplitude: float | None = None,
                 phase: float | None = None) -> None:
        """ Frequency constructor. """
        super().__init__(frequency, phase=phase, amplitude=amplitude)

        # Fundamental frequencies
        self.fundamentals = fundamentals

        # Combination
        self.combination : Combination | None = None
        self.error = 100

    def search(self,
               searchIn: List[FundamentalFrequency],
               minError: float = 1e-15,
               kmax: int = 1,
               nbMinNonZero: int = 0,
               nbMaxNonZero: int = 1,
               withRatio: bool = False,
               mustHaveIndex: List[List[int]] = [],
               isSymetric: bool = False,
               searchResonance: bool = False,
               thresholdResonance: float = np.deg2rad(1),
               verbose: bool = True
               ) -> List[FreqCombination] | None:
        """ Search the best set of integers to solve linear combination
            of a subset of the fundamental frequencies to get the 
            current frequency.

            #### PARAMETERS
            ---------------
            * `searchIn` : subset of fundamental frequencies
            * `minError` : accuracy error desired
            * `kmax` : integers  ∈ [-kmax ; kmax]
            * `nbMinNonZero` : minimum number of non zero elements
            * `nbMaxNonZero` : maximum number of non zero elements
            * `withRatio` : if True, apply a multiplication factor to
            the solution
            * `mustHaveIndex` : list of index compulsatories
            * `isSymetric`: if True, first element is always > 0
        """
        # Frequency to search
        freq = self.freq

        # Candidates for a resonance
        candidates = []

        # Starting from nbMin to nbMax non-zero terms
        for nbNonZero in range(nbMinNonZero, nbMaxNonZero + 1):
            # Generator of possible combinations
            generator = CoeffGenerator(
                len(searchIn),
                nbNonZero,
                kmax,
                withRatio,
                mustHaveIndex,
                isSymetric
            )

            for iter in generator:
                value = np.dot(np.array(searchIn), np.array(iter))

                if withRatio and freq and value:
                    factor = freq // value
                else:
                    factor = 1
                
                error = freq - factor * value

                # Show solution
                if searchResonance:
                    if np.abs(error) < thresholdResonance:
                        coefficients = [
                            int(factor * element) 
                            for element in iter
                        ]
                        if np.sum(coefficients) == 0:

                            goodCombi = FreqCombination(
                                coefficients,
                                searchIn
                            )

                            if verbose:
                                print(f"{str(goodCombi):<50} | {goodCombi.value:9.6f}")
                            candidates.append(goodCombi)

                # Solution with minimum error
                if np.abs(error) <= self.error:
                    
                    self.error = np.abs(error)
                    coefficients = [
                        int(factor * element) 
                        for element in iter
                    ]

                    self.combination = FreqCombination(
                        coefficients,
                        searchIn
                    )
                    
                    # Stop the search if error is small enough
                    if self.error < minError and not searchResonance:
                        return
            
        if searchResonance:
            return candidates

    @staticmethod
    def load(filename: str, 
             fundamentals: List[FundamentalFrequency]
             ) -> List[Frequency]:
        """ Read frequencies from a file. """
        # Read file
        data = pd.read_csv(
            filename, sep='\s+', header=None, comment='#',
            names=['id', 'frequency', 'amplitude', 'phase', 'period']
        )

        # List of Frequency
        _list = []
        for _, row in data.iterrows():
            _list.append(
                Frequency(row['frequency'],
                          fundamentals,
                          row['amplitude'],
                          row['phase'])
            )
        
        return _list

    @staticmethod
    def save(
        frequencies: List[Frequency],
        filename: str,
        minError: float = 1e-7,
        header: str | None = None
        ) -> None:

        with open(filename, 'w') as file:
            if header:
                file.write(f"{header}\n")

            file.write((
                f"{'':<2}    "
                f"{'amplitude':^22}  "
                f"{'combination':^22}  "
                f"{'error':^13}  "
                f"{'fréquence':^18}  "
                f"{'période':^18}"
                f"\n"
            ))

            for ii, freq in enumerate(frequencies):
                file.write((
                    f"{ii:<2}    "
                    f"{freq.amp:<22.15f}  "
                    f"{str(freq.combination) if freq.error < minError else '':<22}  "
                    f"{freq.error if freq.error < minError else 0:13.6e}  "
                    f"{freq.freq:18.15f}  "
                    f"{freq.period:.15f}"
                    f"\n"
                ))

    @staticmethod
    def to_LaTeX(
        frequencies: List[Frequency],
        title: str = "Title",
        label: str = "tab:label") -> str:
        """ Produce a LaTeX table for ephemerides. """
        # Header and footer
        header = rf"""
        \begin{{table}}[h]
        \centering
        \begin{{tabular}}{{ccclc}}
        \toprule
        \multicolumn{{5}}{{c}}{{{title}}} \\
        \midrule
        n° & amplitude ($km$) & fréquence & identification & erreur \\
        \midrule"""

        footer = rf"""
        \bottomrule
        \end{{tabular}}
        \label{{{label}}}
        \end{{table}}
        """

        # Frequencies
        main = "\n".join([(
            rf"{ii} & {freq.amp:10f} & {freq.freq:.12f} & "
            rf"{freq.combination.LaTeX} & {freq.error}") 
            for ii, freq in enumerate(frequencies)
        ])

        return (
            f"{cleandoc(header)}\n"
            f"{main}\n"
            f"{cleandoc(footer)}"
        )

class FundamentalFrequency(BaseFrequency):
    """ Fundamental frequency class. """
    def __init__(self,
                 name: str,
                 frequency: float,
                 phase: float) -> None:
        """ Fundamental frequency constructor. """
        self.name = name
        super().__init__(frequency, phase=phase)

    @property
    def body(self) -> str:
        """ Get body name associated with the Frequency. """
        key = str(self).split('_')[-1]

        if key == 'S':          return 'styx'
        elif key == 'N':        return 'nix'
        elif key == 'K':        return 'kerberos'
        elif key == 'H':        return 'hydra'
        else:                   return 'charon'
    
    @property
    def angle(self) -> str:
        """ Get element key string associated with the Frequency. """
        key = str(self).split('_')[0]

        if key == 'n':      return 'long'
        else:               return key

    @staticmethod
    def load(filename: str) -> List[FundamentalFrequency]:
        """ Read fundamental frequencies from a file."""
        # DataFrame
        data = pd.read_csv(
            filename, sep='\s+', header=None, skiprows=1, comment='#',
            names=['id', 'frequency', 'phase', 'weight', 'name']
        ).astype({'frequency': float, 'phase': float})

        # List of Fundamental Frequencies
        _list = []
        for _, row in data.iterrows():
            _list.append(
                FundamentalFrequency(row['name'], 
                                     row['frequency'],
                                     row['phase']))
        
        return _list

    # ALgebric operations

    def __mul__(self, other: float) -> float:
        return self.freq * other
    
    def __rmul__(self, other: float) -> float:
        return self.freq * other

    def __add__(self, other: FundamentalFrequency | float) -> float:
        if type(other) == FundamentalFrequency:
            return self.freq + other.freq
        else:
            return self.freq + other
        
    def __radd__(self, other: FundamentalFrequency | float) -> float:
        return self + other
    
    def __sub__(self, other: FundamentalFrequency | float) -> float:
        if type(other) == FundamentalFrequency:
            return self.freq - other.freq
        else:
            return self.freq - other
    
    def __rsub__(self, other: FundamentalFrequency | float) -> float:
        return self - other
    
    def __floordiv__(self, other: float) -> int:
        return int(self.freq // other)

    def __str__(self) -> str:
        """ Name of the Fundamental Frequency. """
        return self.name

class LeePeale():
    """ Fundamental frequencies of a moon orbiting around a binary
        system as described in:
        [On the orbits and masses of the satellites of the Pluto-Charon 
        system, Lee & Peale, Icarus 184 (2006) 573-583]
    """
    def __init__(self,
                 massPrimary,
                 massSecundary,
                 separation,
                 G_param : float = 6.67430e-20
                 ) -> None:
        """ Lee & Peale constructor. """
        self.G = G_param

        # Binary parameters
        self.mp   = massPrimary
        self.ms   = massSecundary
        self.Mtot = massPrimary + massSecundary
        self.Mred = massPrimary * massSecundary / self.Mtot
        self.apc  = separation
        self.npc  = self.__meanMotion_kepler(separation)

    def __mass_exp(self, k: int) -> float:
        """ Factor M = (Mp**k + Ms**k) / (Mp + Ms)**k """
        return (self.mp / self.Mtot)**k + (self.ms / self.Mtot)**k
    
    def nk(self, dist):
        nK = (self.G * self.Mtot / dist**3)**0.5 
        return nK * 86400
        
    def __meanMotion_kepler(self, radius: float, inDays=True) -> float:
        """ Keplerian mean motion at r = `radius`. """
        nK = (self.G * self.Mtot / radius**3)**0.5 

        if inDays:
            return nK * 86400
        else:
            return nK
    
    def n1(self, Rs):
        G  = self.G
        M  = self.Mtot
        mu = self.Mred
        M3 = (self.mp / M)**3 + (self.ms / M)**3
        M5 = (self.mp / M)**5 + (self.ms / M)**5
        rat = self.apc / Rs

        return (G * M / Rs**3 * (1 + mu / M * (3/4 * rat**2 + 45/64 * M3 * rat**4 + 175/256 * M5 * rat**6)))**0.5 * 86400

    def k1(self, Rs):
        G  = self.G
        M  = self.Mtot
        mu = self.Mred
        M3 = (self.mp / M)**3 + (self.ms / M)**3
        M5 = (self.mp / M)**5 + (self.ms / M)**5
        rat = self.apc / Rs

        return (G * M / Rs**3 * (1 - mu / M * (3/4 * rat**2 + 135/64 * M3 * rat**4 + 875/256 * M5 * rat**6)))**0.5 * 86400
    
    def v1(self, Rs):
        G  = self.G
        M  = self.Mtot
        mu = self.Mred
        M3 = (self.mp / M)**3 + (self.ms / M)**3
        M5 = (self.mp / M)**5 + (self.ms / M)**5
        rat = self.apc / Rs

        return (G * M / Rs**3 * (1 + mu / M * (9/4 * rat**2 + 225/64 * M3 * rat**4 + 1225/256 * M5 * rat**6)))**0.5 * 86400
    
    def n(
        self, 
        radius: float,
        inDays: bool = True,
        name: str = None
        ) -> FundamentalFrequency:
        """ Mean motion frequency. """
        ratio = self.apc / radius
        value = 1 + self.Mred / self.Mtot * (
                      3/4 * ratio**2
                +   45/64 * self.__mass_exp(3) * ratio**4
                + 175/256 * self.__mass_exp(5) * ratio**6
                )
        
        freq = np.sqrt(self.G * self.Mtot / radius**3 * value)
        
        if inDays:
            freq *= 86400

        if name is None:
            name = "n"

        return FundamentalFrequency(name, freq, 0.)

    def k(
        self, 
        radius: float, 
        inDays: bool = True
        ) -> FundamentalFrequency:
        """ Epicyclic frequency. """
        ratio = self.apc / radius
        value = 1 - self.Mred / self.Mtot * (
                      3/4 * ratio**2
                +  135/64 * self.__mass_exp(3) * ratio**4
                + 875/256 * self.__mass_exp(5) * ratio**6
                )

        freq = np.sqrt(self.G * self.Mtot / radius**3 * value)
        
        if inDays:
            freq *= 86400

        return FundamentalFrequency("k", freq, 0.)
    
    def v(
        self, 
        radius: float, 
        inDays: bool = True
        ) -> FundamentalFrequency:
        """ Vertical frequency. """
        ratio = self.apc / radius
        value = 1 + self.Mred / self.Mtot * (
                      9/4  * ratio**2
                +  225/64  * self.__mass_exp(3) * ratio**4
                + 1225/256 * self.__mass_exp(5) * ratio**6
                )
        
        freq = np.sqrt(self.G * self.Mtot / radius**3 * value)
        
        if inDays:
            freq *= 86400

        return FundamentalFrequency("v", freq, 0.)
    
    def pi(
        self, 
        radius: float, 
        inDays=True
        ) -> FundamentalFrequency:
        """ Periapse precession rate, \varpi = n - \kappa. """
        n = self.n(radius, inDays)
        k = self.k(radius, inDays)

        return FundamentalFrequency("pi", n.freq - k.freq, 0.)
    
    def Omega(
        self, 
        radius: float, 
        inDays=True
        ) -> FundamentalFrequency:
        """ Nodal precession rate, \Omega = n - \nu. """
        n = self.n(radius, inDays)
        v = self.v(radius, inDays)

        return FundamentalFrequency("Om", n.freq - v.freq, 0.)
