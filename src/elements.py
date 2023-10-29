""" Base class for orbital elements. 
"""
from __future__ import annotations
from typing import Tuple
from .vectors import _3Vector, Position, Velocity
import numpy as np

class Elements():
    """ Orbital elements class."""
    eps = 1e-15

    def __init__(self,
                 a: np.ndarray,
                 z: np.ndarray,
                 zeta: np.ndarray,
                 long: np.ndarray,
                 mu: float
                 ) -> None:
        """ Lagrangian orbital elements constructor.
            *a    : semi major axis
            *z    : `e exp(sqrt(-1) * omega)`
            *zeta : `2 sin(i/2) exp(sqrt(-1) Omega)`
            *long : `Omega + omega + M` ie true longitude
            *mu   : standard gravitational parameter
        """
        self.a    = a
        self.z    = z
        self.zeta = zeta
        self.long = long
        self.mu   = mu

    @property
    def lagrangian(self) -> Tuple[np.ndarray, ...]:
        """ Returns the Lagrangian elements set.
            (a, z, zeta, exp(sqrt(-1) * long))
        """
        return (self.a, self.z, self.zeta, self.long)
    
    @property
    def osculating(self) -> Tuple[np.ndarray, ...]:
        """ Returns the osculating elements set.
            (a, e, i, Omega, omega, M)
        """
        k,  h  = np.real(self.z),    np.imag(self.z)
        ix, iy = np.real(self.zeta), np.imag(self.zeta)

        e     = np.sqrt(k**2 + h**2)
        inc   = 2 * np.arcsin(((ix**2 + iy**2) / 4)**0.5)
        Omega = np.arctan2(iy / 2 * np.sin(inc / 2),
                           ix / 2 * np.sin(inc / 2))
        pi    = np.arctan2(h / e, k / e)
        omega = (pi - Omega) % (2. * np.pi)
        M     = (self.long - omega - Omega) % (2. * np.pi)
        
        return (self.a, e, inc, Omega, omega, M)
    
    @property
    def to_vectors(self
        ) -> Tuple(Position, Velocity):
        """ Returns state vectors (Position and Velocity) from
            elements.
            The method is based on [An analytical solution for Kepler's 
            problem - Andras Pál, MNRAS 396, 1737-1742, 2009]
        """
        # 6 parameters set
        a = self.a
        k, h = np.real(self.z), np.imag(self.z)
        q, p = np.real(self.zeta / 2.), np.imag(self.zeta / 2.)
        l = self.long
        
        # Eccentric longitude F
        F = l - k * np.sin(l) + h * np.cos(l)
        while True:
            corr = ( (l - F + k * np.sin(F) - h * np.cos(F)) 
                  / (1 - k * np.cos(F) - h * np.sin(F)))
            F += corr

            if np.all(np.abs(corr) < self.eps):
                break
        
        # Perifocal frame
        Phi = np.sqrt(1 - k**2 - h**2)
        Psi = 1 / (1 + Phi)

        X = a * (np.cos(F) - k - Psi * h * (l - F))
        Y = a * (np.sin(F) - h + Psi * k * (l - F))

        n   = np.sqrt(self.mu / a**3)
        r_a = 1 - k * np.cos(F) - h * np.sin(F)

        Vx = n * a / r_a * (- np.sin(F) + Psi * h * (1 - r_a))
        Vy = n * a / r_a * (  np.cos(F) - Psi * k * (1 - r_a))

        # Rotation
        Xi = np.sqrt(1 - q**2 - p**2)

        pos = Position(
            x = (1 - 2 * p**2) * X + (2 * p * q)    * Y,
            y = (2 * p * q)    * X + (1 - 2 * q**2) * Y,
            z = (-2 * p * Xi)  * X + (2 * q * Xi)   * Y 
        )
        vel = Velocity(
            x = (1 - 2 * p**2) * Vx + (2 * p * q)    * Vy,
            y = (2 * p * q)    * Vx + (1 - 2 * q**2) * Vy,
            z = (-2 * p * Xi)  * Vx + (2 * q * Xi)   * Vy 
        )

        return pos, vel

    @staticmethod
    def from_vectors(r: Position, v: Velocity, mu:float, verbose=False) -> Elements:
        """ Compute orbital elements from cartesian vectors, given
            the standard gravitational parameter of the system.
            /!\ The coordinates system shall be inertial ?
            
            The method is based on the definition given by Pál:
            [An analytical solution for Kepler's problem - Andras Pál,
            MNRAS 396, 1737-1742, 2009]
        """
        # Cross and dot product
        rv  = _3Vector.cross_product(r, v)
        dot = _3Vector.dot_product(r, v)

        # Lagrangian orbital elements (k, h, ix, iy)
        ix = - np.sqrt(2 / (1 + rv.z / rv.norm)) / rv.norm * rv.y
        iy =   np.sqrt(2 / (1 + rv.z / rv.norm)) / rv.norm * rv.x
        k  = (rv.norm / mu * ( v.y - v.z / (rv.norm + rv.z) * rv.y)
              - 1 / r.norm * ( r.x - r.z / (rv.norm + rv.z) * rv.x))
        h  = (rv.norm / mu * (-v.x + v.z / (rv.norm + rv.z) * rv.x)
              - 1 / r.norm * ( r.y - r.z / (rv.norm + rv.z) * rv.y))
        
        # Semi major axis
        a = rv.norm2 / (mu * (1 - k**2 - h**2))

        if verbose:
            print(v[0], v.norm[0])
            print(r[0], r.norm[0])
            print([el / 86400 for el in v[0]], v.norm[0] / 86400)
            print(mu / 86400**2)
            print("sma = ", a[0])

        # True longitude
        l = 1 - np.sqrt(1 - k**2 - h**2)
        long = (np.arctan2(
                    (   - r.norm * v.x 
                        + r.norm * v.z * rv.x / (rv.norm + rv.z) 
                        - k * dot / (2 - l)),
                    (     r.norm * v.y
                        - r.norm * v.z * rv.y / (rv.norm + rv.z)
                        + h * dot / (2 - l))) 
                - dot / rv.norm * (1 - l))

        # Complex elements
        z    =  k + 1j *  h
        zeta = ix + 1j * iy

        return Elements(a, z, zeta, long, mu)

    def __getitem__(self, key: str) -> np.ndarray:
        """ Access an elements through its string representation."""
        # Lagrangian elements
        if key in ['a', 'z', 'zeta', 'long']:
            a, z, zeta, long = self.lagrangian

            if key == 'a':          return a
            elif key == 'z':        return z
            elif key == 'zeta':     return zeta
            elif key == 'long':     return long
        
        # Osculating elements
        elif key in ['a', 
                     'e', 
                     'i', 'inc', 
                     'Omega', 'Om', 
                     'omega', 'om', 
                     'M', 
                     'pi']:
            a, e, inc, Om, om, M = self.osculating

            if key == 'a':                  return a
            elif key =='e':                 return e
            elif key in ['i', 'inc']:       return inc
            elif key in ['Omega', 'Om']:    return Om
            elif key in ['omega', 'om']:    return om
            elif key == 'M':                return M
            elif key == 'pi':               return (Om + om) % (2 * np.pi)
        
        else:
            raise KeyError

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        # Lagrangian elements
        if key in ['a', 'z', 'zeta', 'long']:
            if key == 'a':          self.a = value
            elif key == 'z':        self.z = value
            elif key == 'zeta':     self.zeta = value
            elif key == 'long':     self.long = value

        else:
            raise KeyError
