from curobo._src.optim._portable import PortableOptimizer
class GradientOptCore(PortableOptimizer):
    def finish_init(self): return self
    @property
    def horizon(self): return self.action_horizon
    @property
    def solver_names(self): return [self.config.solver_name]
    def get_all_rollout_instances(self): return self._rollout_list
    def compute_metrics(self,action):
        fn=getattr(self.rollout_fn,"compute_metrics",None); return fn(action) if fn else None
    def reset_shape(self): self.reset()
    def reset_seed(self): self.reset()
    def reset_cuda_graph(self): self.reset()
    def get_recorded_trace(self): return self.debug
    def update_niters(self,niters): self.config.update_niters(niters)
    def update_solver_params(self,solver_params):
        for k,v in solver_params.get(self.config.solver_name,{}).items(): setattr(self.config,k,v)
    def update_goal_dt(self, goal_dt):
        return super().update_goal_dt(goal_dt)
    def debug_dump(self,file_path=""): raise NotImplementedError("debug serialization is not part of the portable optimizer surface")
