from curobo._src.rollout.metrics import CostCollection
from .cost_manager_robot_cfg import RobotCostManagerCfg
class RobotCostManager:
    def __init__(self, device_cfg=None): self.device_cfg=device_cfg; self.costs={}
    def register_cost(self,name,component): self.costs[name]=component
    def get_cost(self,name): return self.costs[name]
    def has_cost(self,name): return name in self.costs
    def enable_cost_component(self,name): self.costs[name].enable_cost()
    def disable_cost_component(self,name): self.costs[name].disable_cost()
    def get_enabled_costs(self): return [x for x in self.costs.values() if x.enabled]
    def get_cost_component_names(self): return list(self.costs)
    def get_cost_components(self): return list(self.costs.values())
    def setup_batch_tensors(self,batch_size,horizon):
        for x in self.costs.values(): x.setup_batch_tensors(batch_size,horizon)
    def reset(self,reset_problem_ids=None,**kwargs):
        for x in self.costs.values(): x.reset(reset_problem_ids,**kwargs)
    def update_dt(self,dt):
        for x in self.costs.values(): x.update_dt(dt)
    def initialize_from_config(self,config,transition_model=None,scene_collision_checker=None,**kwargs):
        for name in ("self_collision","scene_collision","cspace","start_cspace_dist","target_cspace_dist","tool_pose"):
            cfg=getattr(config,name+"_cfg",None)
            if cfg is not None: self.register_cost(name,cfg.class_type(cfg))
        return self
    def compute_costs(self,state,cost_collection=None,goal=None,**kwargs):
        output=cost_collection or CostCollection()
        for name,cost in self.costs.items():
            try: value=cost(state,**kwargs)
            except TypeError: continue
            if isinstance(value,tuple): value=value[0]
            output.add(value,name)
        return output
    compute_convergence=compute_costs
    def update_params(self,**kwargs): return True
__all__=["RobotCostManager","RobotCostManagerCfg"]
