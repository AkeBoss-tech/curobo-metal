"""Portable particle optimizer public surface.

The distribution and sampler helpers live below this package and are used by
both MPPI and evolution strategies.  Importing those helpers must not eagerly
construct the optimizer import graph: MPPI itself imports the distribution.
Keep the convenience exports lazy so a direct helper import cannot form the
``GaussianDistribution -> particle -> MPPI -> GaussianDistribution`` cycle.
"""

__all__ = ["EvolutionStrategies", "EvolutionStrategiesCfg", "MPPI", "MPPICfg"]


def __getattr__(name: str):
    if name in {"EvolutionStrategies", "EvolutionStrategiesCfg"}:
        from .evolution_strategies import EvolutionStrategies, EvolutionStrategiesCfg

        return {"EvolutionStrategies": EvolutionStrategies, "EvolutionStrategiesCfg": EvolutionStrategiesCfg}[name]
    if name in {"MPPI", "MPPICfg"}:
        from .mppi import MPPI, MPPICfg

        return {"MPPI": MPPI, "MPPICfg": MPPICfg}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
