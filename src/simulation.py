""" Simulation class enhancing the REBOUND python package.
"""
from __future__ import annotations
from typing import Tuple
from tqdm import tqdm
from .system import System
from .vectors import Position, Velocity
import rebound
import numpy as np

class Simulation():
    """ Planetary system simulation class. """
    def __init__(self, 
                 system: System,
                 units: Tuple[str, str, str] = ('km', 'days', 'kg')
                 ) -> None:
        """ Simulation constructor. """
        # System
        self.system = system

        # Create rebound.simulation  instance
        self.simulation = rebound.Simulation()
        self.simulation.units = self.system.units
        self.time = np.array([self.simulation.t])
        
        for body in self.system.bodies.values():
            self.simulation.add(body.particle)
        
        # Units conversion
        self.simulation.convert_particle_units(*units)

        # Transformation to set the reference plane
        if self.system.refPlane:
            p = self.simulation.particles[self.system.refPlane]
            rotation = rebound.Rotation.orbit(
                Omega = p.Omega,
                inc   = p.inc,
                omega = p.omega
            ).inverse()

            self.simulation.rotate(rotation)
    
    def integrate(self, 
                  days: float,
                  steptime: float,
                  verbose: bool = True) -> None:
        """ Integrate the system forward in time. """
        # Parameters of integration
        self.simulation.integrator = 'ias15'
        tIni = self.simulation.t
        self.time = np.arange(tIni, tIni + days + steptime, steptime)
        size = self.time.shape[0]

        # Initialize vectors
        for name in self.system.bodies.keys():
            self.system.bodies[name].init_vectors(size)
        
        # Integration
        if verbose:
            print((
                f"Integration from t = {tIni} days "
                f"to t = {tIni + days} days.\n"
                f"N = {size}"
                ))
        
        self.simulation.move_to_com()
        particles = self.simulation.particles
        
        for ii in tqdm(range(size)):
            self.simulation.integrate(self.time[ii])

            # Update state vectors
            for kk, body in enumerate(self.system.bodies.values()):
                body.position[ii] = particles[kk].xyz
                body.velocity[ii] = particles[kk].vxyz

    def save(self, filename: str = "simulation.bin") -> None:
        """ Save simulation as a *.bin file. """
        self.simulation.save(filename)
        self.system.save(filename.replace('.bin', '.syst'))

    @staticmethod
    def load(savebin: str, snapshot: int = 0) -> Simulation:
        """ Load simulation *.bin file. """
        # Load System() with *.syst
        syst = System.load(savebin.replace('.bin', '.syst'))

        # Load *.bin
        sim = Simulation(syst)
        sim.simulation = rebound.Simulation(savebin, snapshot)

        return sim
