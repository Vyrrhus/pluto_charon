""" Base class representing a planetary system of multiple bodies.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Literal
from .vectors import Position, Velocity
from .body import Body
import numpy as np
import pandas as pd
from .utils import G_PARAM_SI

class System():
    """ Hierarchical planetary system class. """
    def __init__(self, 
                 bodies: Dict[str, Body],
                 isBarycenter: List[str] = '',
                 refPlane: str | None = None,
                 units: Tuple[str, str, str] = ('km', 's', 'kg')
                 ) -> None:
        """ System constructor from a list of bodies, from inner
            to outer body. 
        """
        # Bodies & barycenter
        self.bodies = bodies
        if isBarycenter == '':  self.isBarycenter = [bodies.keys()[0]]
        else:                   self.isBarycenter = isBarycenter

        # Ecliptic derived from a reference plane
        self.refPlane = refPlane
        
        # Units (length, time, mass)
        self.units = units

    @property
    def Mtot(self) -> float:
        """ Total mass of the system. """
        return np.sum([el.mass for el in self.bodies.values()])
    
    @property
    def order(self) -> List[str]:
        """ Sorted list of bodies from nearest to farthest of barycenter """
        return sorted(
            self.bodies, 
            key=lambda k: np.mean(self.bodies[k].cylindrical(self)[0])
        )
    
    @property
    def CoM(self) -> Position:
        """System's Center of Mass position. """
        size = next(iter(self.bodies.values())).position.size
        return sum(
            [el.mass * el.position for el in self.bodies.values()],
            start = Position.zeros(size)
            ) / self.Mtot
    
    @property
    def vCoM(self) -> Velocity:
        """ System's Center of Mass velocity. """
        size = next(iter(self.bodies.values())).velocity.size
        return sum(
            [el.mass * el.velocity for el in self.bodies.values()],
            start = Velocity.zeros(size)
            ) / self.Mtot
    
    def save(self, filename: str = "system.syst") -> None:
        """ Save system as a *.syst file. """
        # Data
        data = {
            name: [
                body.particle.m,
                body.particle.x,  body.particle.y,  body.particle.z,
                body.particle.vx, body.particle.vy, body.particle.vz,
                name in self.isBarycenter,
                name == self.refPlane,
                self.units
            ] for name, body in self.bodies.items()
        }

        # Save dataFrame
        pd.DataFrame.from_dict(
            data, 
            orient='index', 
            columns=[
                'm', 
                'x', 'y', 'z',
                'vx', 'vy', 'vz',
                'isBarycenter', 'isRefPlane', 'units']
            ).to_csv(filename)

    def shift_system(
            self, 
            moon: str | None,
            delta: float,
            kind: Literal['GM', 'SMA'] = 'SMA',
            G: float = G_PARAM_SI
        ) -> System:
        """ Returns a new System with [moon]'s SMA shifted by [delta]"""

        current_bodies = {
            name: Body(**dict(
                GM=body.GM,
                x=body.particle.x, vx=body.particle.vx,
                y=body.particle.y, vy=body.particle.vy,
                z=body.particle.z, vz=body.particle.vz,
                hash=name
                ))
            for name, body in self.bodies.items()
        }
        if moon is None:
            return current_bodies
        
        body = self.bodies[moon]
        
        if kind == 'SMA':
            # Compute SMA from state vectors
            elements = body.elements(self, G)
            elements.a += delta

            # Set the new state vectors for the moon
            pos, vel = elements.to_vectors
            new_body = Body(**dict(
                GM=body.GM,
                x=pos.x, vx=vel.x,
                y=pos.y, vy=vel.y,
                z=pos.z, vz=vel.z,
                hash=moon
                ))
        
        elif kind == 'GM':
            new_body = Body(**dict(
                GM=body.GM + delta,
                x=body.particle.x, vx=body.particle.vx,
                y=body.particle.y, vy=body.particle.vy,
                z=body.particle.z, vz=body.particle.vz,
                hash=moon
                ))
        
        current_bodies[moon] = new_body

        return System(
            current_bodies, 
            self.isBarycenter, 
            self.refPlane, 
            self.units)


    @staticmethod
    def load(savesyst: str) -> System:
        """ Load system *.syst file. """
        # Load file
        dataFrame = pd.read_csv(savesyst, index_col=0)
        data = dataFrame.to_dict(orient='index')

        # Set barycenter, reference plane and units
        isBarycenter = dataFrame[
            dataFrame['isBarycenter']].index.values.to_list()
        refPlane = ''.join(
            dataFrame[dataFrame['isRefPlane']].index.values)
        units = eval(dataFrame['units'].iloc[0])

        # Build System()
        return System(
            {name: Body(**kwargs) for name, kwargs in data.items()},
            isBarycenter = isBarycenter,
            refPlane = refPlane,
            units = units
        )

