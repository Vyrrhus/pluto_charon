""" Base class for _3Vector, Position, Velocity.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np

class _3Vector:
    """ Cartesian vector class. """
    def __init__(self,
                 x: np.ndarray = np.zeros(0),
                 y: np.ndarray = np.zeros(0),
                 z: np.ndarray = np.zeros(0)) -> None:
        """ 3D vector constructor"""
        self.x = x
        self.y = y
        self.z = z
    
    @property
    def size(self) -> int:
        """ Number of vectors. """
        return self.x.shape[0]
    
    @property
    def norm2(self) -> np.ndarray:
        """ 2-norm squared of each vectors. """
        return self.x**2 + self.y**2 + self.z**2
    
    @property
    def norm(self) -> np.ndarray:
        """ 2-norm of each vectors. """
        return np.sqrt(self.norm2)
    
    @property
    def numpy(self) -> np.ndarray:
        """ Numpy array object equivalent"""
        return np.array([self.x, self.y, self.z])

    @staticmethod
    def zeros(size: int) -> _3Vector:
        """ Create a _3Vector with given size. """
        return _3Vector(np.zeros(size),
                        np.zeros(size),
                        np.zeros(size))
    
    @staticmethod
    def cross_product(v1: _3Vector, v2: _3Vector) -> _3Vector:
        """ Returns the cross product of two _3Vectors."""
        x = v1.y * v2.z - v1.z * v2.y
        y = v1.z * v2.x - v1.x * v2.z
        z = v1.x * v2.y - v1.y * v2.x
        return _3Vector(x, y, z)

    @staticmethod
    def dot_product(v1: _3Vector, v2: _3Vector) -> np.ndarray:
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

    def __getitem__(self, index: int) -> Tuple:
        return self.x[index], self.y[index], self.z[index]
    
    def __setitem__(self, 
                    index:int, 
                    values: Tuple[float, float, float]) -> None:
        self.x[index] = values[0]
        self.y[index] = values[1]
        self.z[index] = values[2]

    def __add__(self, o: _3Vector) -> _3Vector:
        return _3Vector(x = self.x + o.x,
                        y = self.y + o.y,
                        z = self.z + o.z)
    
    def __sub__(self, o: _3Vector) -> _3Vector:
        return _3Vector(x = self.x - o.x,
                        y = self.y - o.y,
                        z = self.z - o.z)
    
    def __mul__(self, scale: float) -> _3Vector:
        return _3Vector(x = self.x * scale,
                        y = self.y * scale,
                        z = self.z * scale)
    
    def __rmul__(self, scale: float) -> _3Vector:
        return _3Vector(x = self.x * scale,
                        y = self.y * scale,
                        z = self.z * scale)
    
    def __truediv__(self, scale: float) -> _3Vector:
        return _3Vector(x = self.x / scale,
                        y = self.y / scale,
                        z = self.z / scale)

    def __rtruediv__(self, scale: float) -> None:
        raise TypeError

    def __len__(self) -> int:
        return self.size
    
    def __str__(self) -> str:
        return "\n".join(
            [" ".join(
                f"{el:.6f}" for el in row)
            for row in np.column_stack(
                                (self.x, self.y, self.z))
            ])
    
class Position(_3Vector):
    """ Position vector class inherited from _3Vector. """
    pass

class Velocity(_3Vector):
    """ Velocity vector class inherited from _3Vector. """
    pass

