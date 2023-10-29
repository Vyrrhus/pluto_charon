# Pluto-Charon analysis

Using the modules provided within this repository, one can simulate the orbits of a hierarchichal system of objects, transform the coordinates obtained and perform a frequency analysis.

## Use
To install all the packages within a virtual environment, one can use

    pip install -r requirements.txt

Orbit propagation relies on the [REBOUND library](https://github.com/hannorein/rebound), using [IAS15 integrator](https://academic.oup.com/mnras/article/446/2/1424/2892331) although others could be used as well given the use case.

## Initial conditions for Pluto-Charon
Within `src/data.py`, two sets of initial conditions are provided for each body of the Pluto-Charon system.
The first one can be found using [JPL Horizons tool](https://ssd.jpl.nasa.gov/horizons/app.html#/) (see the input parameters used within the file itself). The other one is extracted from a kernel provided [here](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/) which uses a more recent solution (PLU58).

## Easy start : simulation & FFT
Propagate initial conditions provided within `src/data.py`:
    from src.simulation import Simulation
    from src.utils import set_system

    # Simulate Pluto, Charon and Hydra for 1000 days with a 1-day time step
    system = set_system(
        ["pluto", "charon", "hydra"],
        barycenter=["pluto", "charon"],
        refPlane="charon"
    )
    simulation = Simulation(system)
    simulation.integrate(1000, 1)

Get cylindrical coordinates for Hydra and classic orbital elements for Charon

    hydra = simulation.system.bodies["hydra"]
    r, theta, z = hydra.cylindrical(simulation.system)

    charon = simulation.system.bodies["charon"]
    charon_elements = charon.elements(simulation.system, simulation.simulation.G)
    
Perform a FFT on Hydra's semi-major axis

    fftfreqs, fft = hydra.fft(simulation, "a")

## Frequency analysis
Using [Laskar's Frequency Map Analysis](https://link.springer.com/chapter/10.1007/978-94-011-4673-9_13) (a FORTRAN software was used but it is not provided here), one can get the main frequencies of any signal with great accuracy.

`src/frequency.py` helps to read the outputs of that tool, find a linear integer combination of fundamental frequencies to produce ephemerides, and compute the frequencies obtained by the analytical model of a near-circular, near-equatorial circumbinary orbit presented by [Lee and Peale](https://doi.org/10.1016/j.icarus.2006.04.017).