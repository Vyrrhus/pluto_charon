""" Miscellaneous functions. """
from __future__ import annotations
from typing import List, Literal, Dict, TYPE_CHECKING
from time import time
import numpy as np
import subprocess
import os

from src import data

if TYPE_CHECKING:
    from .body import Body
    from .system import System
    from .vectors import Position

G_PARAM_SI = 6.67430e-20

def timer(func):
    """Get runtime of decorated function"""
    def wrapper(*args, **kwargs):
        tIni = time()
        result = func(*args, **kwargs)
        tEnd = time()
        print(
            f"Function <{func.__name__}> executed in {tEnd-tIni:.4f}s")
        return result
    return wrapper

def FMA(
        y: np.ndarray,
        t: np.ndarray,
        filename: str,
        Nfreq: int,
        inputPath:  str = 'data/FMA/input.dat',
        outputPath: str = 'data/FMA/output.dat',
        exePath:    str = 'frequency_analysis',
        timeout: int | None = None,
        verbose: bool = True
        ) -> None:
    """ Execute Frequency Map Analysis on `y(t)`. 
        Laskar, 1999.
    """
    # Input file
    with open(inputPath, 'w') as file:
        # Header
        file.write(f"{t.shape[0]} {t[0]:.15f} {t[1] - t[0]:.15f}\n")

        # Data
        np.savetxt(
            file,
            np.column_stack((np.real(y), np.imag(y))),
            fmt='%.15f')
    
    # Execute FORTRAN program
    kwargs = {
        "timeout": timeout,
        "stdout": None if verbose else subprocess.DEVNULL,
        "stderr": None if verbose else subprocess.STDOUT
        }
    
    subprocess.run([exePath] + [str(Nfreq)], **kwargs)
    os.rename(outputPath, filename)

def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

def set_system(
        moons: List[Literal["styx", "nix", "kerberos", "hydra"]],
        barycenter: List[str] = ["pluto", "charon"],
        refPlane: str = "charon",
        **kwargs
    ) -> System:
    """ Returns a Pluto Charon system with 2 central bodies + moons 
        The system can be shifted from its original position using the
        kwargs `sma` and `inc`.
        ------
        sma: Dict[str, float]: adds $\Delta a$ to the semi major axis of
                               a given body.

        inc: Dict[str, float]: sets the orbit's inclination of a given
                               body.
    """
    bodies = {
        "pluto":    Body(**data.pluto),
        "charon":   Body(**data.charon),
        "styx":     Body(**data.styx),
        "nix":      Body(**data.nix),
        "kerberos": Body(**data.kerberos),
        "hydra":    Body(**data.hydra)
    }

    # Change the system
    if "sma" in kwargs or "inc" in kwargs:
        # Get initial position & velocity
        syst = System(
            {
                name: body
                for name, body in bodies.items()
                if name in ["pluto", "charon"] + moons
            },
            isBarycenter=barycenter,
            refPlane=refPlane
        )

        for body in syst.bodies.values():
            p = body.particle
            pos = [np.array([el]) for el in [p.x, p.y, p.z]]
            vel = [np.array([el]) for el in [p.vx, p.vy, p.vz]]
            body.position = Position(*pos)
            body.velocity = Position(*vel)

        # SMA
        sma: Dict[str, float] = kwargs["sma"] if "sma" in kwargs.keys() else {}
        inc: Dict[str, float] = kwargs["inc"] if "inc" in kwargs.keys() else {}
        for name in bodies.keys():
            if name not in sma.keys() and name not in inc.keys():
                continue

            sat = syst.bodies[name]
            elems    = sat.elements(syst, G_PARAM_SI)
            if name in sma.keys():
                elems.a += sma[name]
            
            if name in inc.keys():
                _, _, _, Om, _, _ = elems.osculating
                elems.zeta = 2 * np.sin(inc[name]/2) * np.exp(1j * Om)

            pos, vel = elems.to_vectors

            bodies[name] = Body(
                GM=sat.mass*G_PARAM_SI,
                x=pos.x, y=pos.y, z=pos.z,
                vx=vel.x, vy=vel.y, vz=vel.z,
                hash=sat.name
            )
        
    return System(
        {
            name: body
            for name, body in bodies.items()
            if name in ["pluto", "charon"] + moons
        },
        isBarycenter=barycenter,
        refPlane=refPlane
    )
