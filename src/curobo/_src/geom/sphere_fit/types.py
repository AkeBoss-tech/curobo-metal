from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import numpy as np
import torch


class SphereFitType(Enum):
    SURFACE = "surface"
    VOXEL = "voxel"
    MORPHIT = "morphit"


@dataclass
class SphereFitMetrics:
    num_spheres: int = 0
    coverage: float = 0.0
    protrusion: float = 0.0
    protrusion_dist_mean: float = 0.0
    protrusion_dist_p95: float = 0.0
    surface_gap_mean: float = 0.0
    surface_gap_p95: float = 0.0
    max_uncovered_gap: float = 0.0
    volume_ratio: float = 0.0


@dataclass
class SphereFitResult:
    centers: torch.Tensor
    radii: torch.Tensor
    num_spheres: int = 0
    metrics: Optional[SphereFitMetrics] = None
    fit_time_s: Optional[float] = None
    used_mesh: Any = field(default=None, repr=False)
    history: List[Tuple[Any, Any]] = field(default_factory=list)
    debug_info: Optional[Dict[str, Any]] = field(default_factory=dict)
