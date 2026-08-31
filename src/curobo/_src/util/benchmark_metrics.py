from dataclasses import dataclass
from typing import Optional,Sequence,List
import numpy as np
def percent_true(arr:Sequence)->float:return 0. if len(arr)==0 else 100.*np.count_nonzero(arr)/len(arr)
@dataclass
class Statistic:
    mean:float;std:float;median:float;percent_25:float;percent_75:float;percent_98:float;min:float;max:float
    @classmethod
    def from_list(cls, lst: Sequence[float]) -> "Statistic":
        a=np.asarray([x for x in lst if x<np.inf],float)
        if not len(a):return cls(*([0.]*8))
        return cls(float(a.mean()),float(a.std()),float(np.median(a)),float(np.percentile(a,25)),float(np.percentile(a,75)),float(np.percentile(a,98)),float(a.min()),float(a.max()))
    def __str__(self) -> str:return f"mean: {self.mean:2.3f} ± {self.std:2.3f}  median: {self.median:2.3f}  75%: {self.percent_75:2.3f}  98%: {self.percent_98:2.3f}"
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
    def from_list(cls, group: List[CuroboMetrics]) -> "CuroboGroupMetrics":
        u=[m for m in group if not m.skip];s=[m for m in u if m.success]
        kw=dict(group_size=len(group),success=percent_true([m.success for m in group]),skips=sum(m.skip for m in group),env_collision_rate=percent_true([m.collision for m in u]),self_collision_rate=percent_true([m.self_collision for m in u]),joint_violation_rate=percent_true([m.joint_limit_violation for m in u]),physical_violation_rate=percent_true([m.physical_violation for m in u]),payload_success=percent_true([m.payload_success for m in group]),perception_success=percent_true([m.perception_success for m in group]),perception_interpolated_success=percent_true([m.perception_interpolated_success for m in group]),within_one_cm_rate=percent_true([m.position_error<1 for m in u]),within_five_cm_rate=percent_true([m.position_error<5 for m in u]),within_fifteen_deg_rate=percent_true([m.orientation_error<15 for m in u]),within_thirty_deg_rate=percent_true([m.orientation_error<30 for m in u]))
        for f in ("eef_position_path_length","eef_orientation_path_length","cspace_path_length","attempts","position_error","orientation_error","motion_time","solve_time","time","perception_time","jerk","energy","torque","power","work","peak_power"):kw[f]=Statistic.from_list([getattr(m,f) for m in s])
        kw["solve_time_per_step"]=Statistic.from_list([m.solve_time/m.trajectory_length for m in s]);return cls(**kw)
    def print_summary(self) -> None:
        print(f"Total problems: {self.group_size}")
        print(f"# Skips (Hard Failures): {self.skips}")
        print(f"% Success: {self.success:4.2f}")
        print(f"% Within 1cm: {self.within_one_cm_rate:4.2f}")
        print(f"% Within 5cm: {self.within_five_cm_rate:4.2f}")
        print(f"% Within 15deg: {self.within_fifteen_deg_rate:4.2f}")
        print(f"% Within 30deg: {self.within_thirty_deg_rate:4.2f}")
        print(f"% With Environment Collision: {self.env_collision_rate:4.2f}")
        print(f"% With Self Collision: {self.self_collision_rate:4.2f}")
        print(f"% With Joint Limit Violations: {self.joint_violation_rate:4.2f}")
        print(f"% With Physical Violations: {self.physical_violation_rate:4.2f}")
        print(f"Eef Position Path Length: {self.eef_position_path_length}")
        print(f"Eef Orientation Path Length: {self.eef_orientation_path_length}")
        print(f"Motion Time: {self.motion_time}")
        print(f"Solve Time: {self.solve_time}")
        print(f"Solve Time Per Step: {self.solve_time_per_step}")
__all__=["Statistic","CuroboMetrics","CuroboGroupMetrics","percent_true"]
