from dataclasses import dataclass
from typing import Optional,Sequence,List
import numpy as np
def percent_true(arr:Sequence)->float:return 0. if len(arr)==0 else 100.*np.count_nonzero(arr)/len(arr)
@dataclass
class Statistic:
    mean:float;std:float;median:float;percent_25:float;percent_75:float;percent_98:float;min:float;max:float
    @classmethod
    def from_list(cls,lst):
        a=np.asarray([x for x in lst if x<np.inf],float)
        if not len(a):return cls(*([0.]*8))
        return cls(float(a.mean()),float(a.std()),float(np.median(a)),float(np.percentile(a,25)),float(np.percentile(a,75)),float(np.percentile(a,98)),float(a.min()),float(a.max()))
    def __str__(self):return f"mean: {self.mean:2.3f} ± {self.std:2.3f}  median: {self.median:2.3f}  75%: {self.percent_75:2.3f}  98%: {self.percent_98:2.3f}"
@dataclass
class CuroboMetrics:
    skip:bool=True;success:bool=False;collision:bool=True;joint_limit_violation:bool=True;self_collision:bool=True;physical_violation:bool=True
    payload_success:bool=False;perception_success:bool=False;perception_interpolated_success:bool=False
    position_error:float=np.inf;orientation_error:float=np.inf;eef_position_path_length:float=np.inf;eef_orientation_path_length:float=np.inf
    cspace_path_length:float=0.;trajectory_length:int=1;attempts:int=1;motion_time:float=np.inf;solve_time:float=np.inf;time:float=np.inf
    perception_time:float=0.;jerk:float=np.inf;energy:float=0.;torque:float=0.;power:float=0.;work:float=0.;peak_power:float=0.
@dataclass
class CuroboGroupMetrics:
    group_size:int=0;success:float=0.;skips:int=0;env_collision_rate:float=0.;self_collision_rate:float=0.;joint_violation_rate:float=0.;physical_violation_rate:float=0.
    payload_success:float=0.;perception_success:float=0.;perception_interpolated_success:float=0.;within_one_cm_rate:float=0.;within_five_cm_rate:float=0.;within_fifteen_deg_rate:float=0.;within_thirty_deg_rate:float=0.
    eef_position_path_length:Optional[Statistic]=None;eef_orientation_path_length:Optional[Statistic]=None;cspace_path_length:Optional[Statistic]=None;attempts:Optional[Statistic]=None
    position_error:Optional[Statistic]=None;orientation_error:Optional[Statistic]=None;motion_time:Optional[Statistic]=None;solve_time:Optional[Statistic]=None;solve_time_per_step:Optional[Statistic]=None
    time:Optional[Statistic]=None;perception_time:Optional[Statistic]=None;jerk:Optional[Statistic]=None;energy:Optional[Statistic]=None;torque:Optional[Statistic]=None;power:Optional[Statistic]=None;work:Optional[Statistic]=None;peak_power:Optional[Statistic]=None
    @classmethod
    def from_list(cls,g:List[CuroboMetrics]):
        u=[m for m in g if not m.skip];s=[m for m in u if m.success]
        kw=dict(group_size=len(g),success=percent_true([m.success for m in g]),skips=sum(m.skip for m in g),env_collision_rate=percent_true([m.collision for m in u]),self_collision_rate=percent_true([m.self_collision for m in u]),joint_violation_rate=percent_true([m.joint_limit_violation for m in u]),physical_violation_rate=percent_true([m.physical_violation for m in u]),payload_success=percent_true([m.payload_success for m in g]),perception_success=percent_true([m.perception_success for m in g]),perception_interpolated_success=percent_true([m.perception_interpolated_success for m in g]),within_one_cm_rate=percent_true([m.position_error<1 for m in u]),within_five_cm_rate=percent_true([m.position_error<5 for m in u]),within_fifteen_deg_rate=percent_true([m.orientation_error<15 for m in u]),within_thirty_deg_rate=percent_true([m.orientation_error<30 for m in u]))
        for f in ("eef_position_path_length","eef_orientation_path_length","cspace_path_length","attempts","position_error","orientation_error","motion_time","solve_time","time","perception_time","jerk","energy","torque","power","work","peak_power"):kw[f]=Statistic.from_list([getattr(m,f) for m in s])
        kw["solve_time_per_step"]=Statistic.from_list([m.solve_time/m.trajectory_length for m in s]);return cls(**kw)
    def print_summary(self):print(f"Total problems: {self.group_size}\n% Success: {self.success:4.2f}")
__all__=["Statistic","CuroboMetrics","CuroboGroupMetrics","percent_true"]
