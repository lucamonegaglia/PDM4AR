<<<<<<< HEAD
from dataclasses import dataclass

@dataclass(frozen=True)
class SatelliteParams():
    orbit_r: float
    omega: float
    tau: float
    radius: float

@dataclass(frozen=True)
class PlanetParams():
    center: list[float, float]
=======
from dataclasses import dataclass

@dataclass(frozen=True)
class SatelliteParams():
    orbit_r: float
    omega: float
    tau: float
    radius: float

@dataclass(frozen=True)
class PlanetParams():
    center: list[float, float]
>>>>>>> ex11/master
    radius: float