from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints, RolloutMetrics, RolloutResult
from curobo._src.state.state_joint import JointState
from .cost_manager.cost_manager_robot import RobotCostManager
from .rollout_robot_cfg import RobotRolloutCfg
class RobotRollout:
    def __init__(self,config=None,scene_collision_checker=None,use_cuda_graph=False):
        self.config=config; self._batch_size=1; self.goal=None
        self.cost_manager=RobotCostManager(None if config is None else config.device_cfg)
        if config and config.cost_cfg: self.cost_manager.initialize_from_config(config.cost_cfg,None,scene_collision_checker)
    action_dim=property(lambda self: getattr(self.config.transition_model_cfg,"d_action",0) if self.config else 0)
    action_horizon=property(lambda self: getattr(self.config.transition_model_cfg,"action_horizon",1) if self.config else 1)
    horizon=property(lambda self:self.action_horizon)
    batch_size=property(lambda self:self._batch_size,lambda self,v:setattr(self,"_batch_size",v))
    dt=property(lambda self:getattr(self.config.transition_model_cfg,"dt",1.0) if self.config else 1.0)
    def compute_state_from_action(self,act_seq,**kwargs): return JointState(position=act_seq)
    def evaluate_action(self,act_seq,**kwargs):
        state=self.compute_state_from_action(act_seq); cc=CostsAndConstraints()
        cc.costs=self.cost_manager.compute_costs(state,goal=self.goal,**kwargs)
        return RolloutResult(act_seq,cc,state)
    def compute_metrics_from_state(self,state,**kwargs):
        cc=CostsAndConstraints(); cc.costs=self.cost_manager.compute_costs(state,goal=self.goal,**kwargs)
        return RolloutMetrics(state=state,costs_and_constraints=cc,feasible=True)
    def compute_metrics_from_action(self,act_seq,**kwargs):
        out=self.compute_metrics_from_state(self.compute_state_from_action(act_seq),**kwargs); out.actions=act_seq; return out
    def compute_state_from_action_metrics(self,act_seq,**kwargs):
        state=self.compute_state_from_action(act_seq); return state,self.compute_metrics_from_state(state,**kwargs)
    def update_params(self,goal,num_particles=None): self.goal=goal; return True
    def update_batch_size(self,batch_size): self._batch_size=batch_size
    def update_dt(self,dt): return True
    def reset(self,reset_problem_ids=None,**kwargs): return True
    def reset_shape(self): return True
    def reset_cuda_graph(self): return True
    def reset_seed(self): return None
    def enable_cost_component(self,name): return self.cost_manager.enable_cost_component(name)
    def disable_cost_component(self,name): return self.cost_manager.disable_cost_component(name)
    def get_cost_component_names(self): return self.cost_manager.get_cost_component_names()
    def get_all_cost_components(self): return self.cost_manager.get_cost_components()
    def get_cost_component_by_name(self,name): return self.cost_manager.get_cost(name)
__all__=["RobotRollout","RobotRolloutCfg"]
