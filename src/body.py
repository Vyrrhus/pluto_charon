""" Base class representing an object associated to a
    rebound.Particle instance.
"""
from __future__ import annotations
from typing import Tuple, List, Literal, TYPE_CHECKING
from rebound import Particle
from .vectors import Position, Velocity
from .elements import Elements
from .frequency import FundamentalFrequency, Frequency
from src import utils
import numpy as np
import scipy.fftpack as fftpack
import subprocess
import os
from tqdm import tqdm

if TYPE_CHECKING:
    from .simulation import Simulation
    from .system import System

class Body():
    """ Planetary or small body class. """
    def __init__(self, **kwargs) -> None:
        """ Body constructor, same as rebound.Particle. """
        # Name & GM
        self.name = kwargs.get('hash')
        self.GM   = kwargs.get('GM')

        # Particle representing the current state
        kwargs['m'] = self.GM / utils.G_PARAM_SI
        del kwargs['GM']
        self.particle = Particle(**kwargs)

        # State vectors saved
        self.position = Position(*tuple(map(
                            lambda value: np.array([value]), 
                            self.particle.xyz))
                        )
        self.velocity = Velocity(*tuple(map(
                            lambda value: np.array([value]), 
                            self.particle.vxyz))
                        )

    @property
    def mass(self) -> float:
        """ Body's particle mass"""
        return self.particle.m

    def init_vectors(self, size: int):
        """ Initialize state vectors with a given size. """
        self.position = Position.zeros(size)
        self.velocity = Velocity.zeros(size)

    def cylindrical(
            self, system: System
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Compute cylindrical coordinates (r, theta, z)."""
        # Position relative to the barycenter
        pos = self.position - system.CoM

        # Cylindrical coordinates
        radius = np.sqrt(pos.x**2 + pos.y**2)
        angle  = np.arctan2(pos.y, pos.x)
        z      = pos.z

        return (radius, angle, z)

    def elements(self, system: System, G: float) -> Elements:
        """ Compute orbital elements. """
        # Gravitational standard parameter from HORIZONS (km^3/d^2)
        muHorizons = {
            "charon":       5.1589607879513350E+12,
            "styx":         7.2820772578730361E+12,
            "nix":          7.2820089945584805E+12,
            "kerberos":     7.2820523986696797E+12,
            "hydra":        7.2820053472720469E+12
        }
        # mu = G * system.Mtot
        mu = muHorizons[self.name]

        # Position & velocity relative to the barycenter
        pos = self.position - system.CoM
        vel = self.velocity - system.vCoM

        # Get elements
        elems = Elements.from_vectors(pos, vel, mu)

        return elems

    def fft(self,
            simulation: Simulation,
            key: Literal[
                'r', 'theta', 'rtheta', 'zcyl',
                'Rz',
                'a', 'z', 'zeta', 'long',
                'e', 'i', 'inc', 'Omega', 'Om', 'omega', 'om', 'M', 'pi'
            ]
            ) -> Tuple[np.ndarray, np.ndarray]:
        """ Fast Fourier Transform analysis. """
        # Data to analyze
        data = None
        time = simulation.time
        size = time.shape[0]

        if key in ['r', 'theta', 'rtheta', 'zcyl', 'Rz']:
            r, theta, z = self.cylindrical(simulation.system)
            if key == 'r':          data = r
            elif key == 'theta':    data = theta
            elif key == 'zcyl':     data = z
            elif key == 'rtheta':   data = r * np.exp(1j * theta)
            elif key == 'Rz':       data = np.sqrt(r**2 + z**2)
        
        else:
            elems = self.elements(simulation.system, 
                                  simulation.simulation.G)
            data = elems[key]
        
        # Real --> complex for angles
        if key in ['Omega', 'Om', 'omega', 'om', 'pi', 'theta', 'long']:
            data = np.exp(1j * data)
        
        if data is None:
            print(f"Wrong key : {key} does not exist.")
            exit()
        
        # FFT : real signal
        if key in ['r', 'zcyl', 'a', 'Rz']:
            realFFT = fftpack.rfft(data) / size
            freqs   = fftpack.rfftfreq(
                size,
                (time[-1] - time[0]) / size
            )
            
            return freqs[1:], realFFT[1:]

        # FFT : complex signal
        else:
            complexFFT = 2. * np.abs(fftpack.fft(data)) / size
            freqs      = fftpack.fftfreq(
                size,
                (time[-1] - time[0]) / size
            )
            return freqs[:size//2], complexFFT[:size//2]

    def fma(self,
            simulation: Simulation,
            key: Literal[
                'r', 'theta', 'rtheta', 'zcyl',
                'a', 'z', 'zeta', 'long',
                'e', 'i', 'inc', 'Omega', 'Om', 'omega', 'om', 'M', 'pi'
            ],
            Nfreq: int,
            filename: str,
            timeout: int | None = None,
            verbose: bool = True
            ) -> None:
        """ Frequency Map Analysis [Laskar, 1999]. 
        
            ---------------
            #### Parameters
            * `simulation`: Simulation object
            * `key`: element to analyze
            * `Nfreq`: number of frequencies to search for
            * `filename`: path of the output file
        """
        # Data to analyze
        data = None
        if key in ['r', 'theta', 'rtheta', 'zcyl']:
            r, theta, z = self.cylindrical(simulation.system)
            if key == 'r':          data = r
            elif key == 'theta':    data = theta
            elif key == 'zcyl':     data = z
            elif key == 'rtheta':   data = r * np.exp(1j * theta)
        
        else:
            elems = self.elements(simulation.system, 
                                  simulation.simulation.G)
            data = elems[key]
        
        # Real --> complex for angles
        # if key in ['Omega', 'Om', 'omega', 'om', 'pi', 'theta', 'long']:
        #     data = np.exp(1j * data)
        
        if data is None:
            print(f"Wrong key : {key} does not exist.")
            exit()

        # Run FMA
        utils.FMA(data, simulation.time, filename, Nfreq, 
                  timeout=timeout, verbose=verbose)

    def ephemerides(self,
                    system: System,
                    filename: str,
                    fundamentals: List[FundamentalFrequency],
                    minError: float = 1e-10,
                    nbFreqs: int | None = None,
                    verbose: bool = True
                    ) -> List[Frequency]:
        """ Search the linear combination of `fundamentals` frequencies
            that match each Frequency stored in `filename` with 
            `minError` precision.
            Only the first `nbFreqs` stored are explored.
        """
        # Frequency list
        frequencies = Frequency.load(filename, fundamentals)[:nbFreqs]

        # Print header
        if verbose:
            idx = ' '.join([f"{str(ii+1):<3}" 
                            for ii in range(len(fundamentals))])
            
            print(f"{self.name.capitalize()} : analyzing {filename}")
            print((
                f"\n"
                f"{'Id':^2} | {'Integer combination':^{len(idx)}} | "
                f"{'error':<15} | {'Literal':<30}"
            ))
            print(f"{'':^2} | {idx}")
        

        otherMoons = [
            body for body in system.bodies.values()
            if body.name not in [self.name] + system.isBarycenter
        ]
        
        # Search combinations for each Frequency
        for ii, frequency in enumerate(tqdm(frequencies)):
            
            # 1st step : search within self fundamental frequencies
            searchIn = [
                freq for freq in fundamentals
                if 'PC' in freq.name
                or f"{self.name[0].capitalize()}" in freq.name
            ]
            
            frequency.search(
                searchIn, minError=minError,
                kmax=4, nbMaxNonZero=3, withRatio=True
            )

            # 2nd step : search within other body's
            for body in otherMoons:

                # Stop research
                if frequency.error <= minError:
                    break

                searchIn = [
                    freq for freq in fundamentals
                    if 'PC' in freq.name
                    or f"{body.name[0].capitalize()}" in freq.name
                ]
                
                frequency.search(
                    searchIn, minError=minError,
                    kmax=4, nbMaxNonZero=3, withRatio=True
                )

            # 3rd step : search a combination of self + 1 other body
            for body in otherMoons:

                # Stop research
                if frequency.error <= minError:
                    break

                searchIn = [
                    freq for freq in fundamentals
                    if 'PC' in freq.name
                    or f"{self.name[0].capitalize()}" in freq.name
                    or f"{body.name[0].capitalize()}" in freq.name
                ]
                
                frequency.search(
                    searchIn, minError=minError,
                    kmax=4, nbMaxNonZero=3, withRatio=True,
                    mustHaveIndex=[
                        [ii for ii, el in enumerate(searchIn)
                         if f"{self.name[0].capitalize()}" in el.name],
                        [ii for ii, el in enumerate(searchIn)
                         if f"{body.name[0].capitalize()}" in el.name]
                    ]
                )

            # 4th step : search within all fundamental frequencies
            if frequency.error > minError:
                frequency.search(
                    searchIn, minError=minError,
                    kmax=2, nbMaxNonZero=5, withRatio=True
                )

            # Print
            if verbose:
                print(f"{ii+1:2} | ", end="")
                for value in frequency.combination.coefficients:
                    print(f"{value:<4}", end=" ")
                print((
                    f" | {frequency.error:<15.8e} | "
                    f"{frequency.combination}"
                ))
        
        return frequencies
    