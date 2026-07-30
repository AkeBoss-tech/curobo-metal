from dataclasses import dataclass
from typing import List, Optional

from .solve_mode import SolveMode


@dataclass
class SolveState:
    solve_type: SolveMode
    batch_size: int
    num_envs: int
    num_goalset: int = 1
    multi_env: bool = False
    batch_mode: bool = False
    num_seeds: Optional[int] = None
    num_ik_seeds: Optional[int] = None
    num_graph_seeds: Optional[int] = None
    num_trajopt_seeds: Optional[int] = None
    tool_frames: Optional[List[str]] = None

    def __post_init__(self):
        self.multi_env = self.num_envs != 1
        self.batch_mode = self.batch_size > 1
        if self.num_seeds is None:
            self.num_seeds = self.num_ik_seeds or self.num_trajopt_seeds or self.num_graph_seeds

    def clone(self): return type(self)(**self.__dict__)
    def get_batch_size(self): return self.batch_size * (self.num_seeds or 0)
    def get_ik_batch_size(self): return self.batch_size * (self.num_ik_seeds or 0)
    def get_trajopt_batch_size(self): return self.batch_size * (self.num_trajopt_seeds or 0)


@dataclass
class MotionPlanSolveState:
    solve_type: SolveMode
    ik_solve_state: SolveState
    trajopt_solve_state: SolveState


__all__ = ["SolveState", "MotionPlanSolveState"]
