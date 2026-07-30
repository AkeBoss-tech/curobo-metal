"""Pinned evolution-strategies surface."""

from dataclasses import dataclass

from .mppi import MPPI, MPPICfg


@dataclass
class EvolutionStrategiesCfg(MPPICfg):
    solver_type: str = "es"
    solver_name: str = "es"
    learning_rate: float = 0.1


class EvolutionStrategies(MPPI):
    strategy = "cem"


__all__ = ["EvolutionStrategiesCfg", "EvolutionStrategies"]
