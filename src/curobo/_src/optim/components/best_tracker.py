import torch
from curobo._src.optim.gradient.update_best_solution import update_best_solution
class BestTracker:
    def __init__(self,*args,**kwargs): self.best_action=self.best_cost=None
    def resize(self,num_problems,action_horizon,action_dim): self.best_action=torch.zeros(num_problems,action_horizon,action_dim); self.best_cost=torch.full((num_problems,),torch.inf)
    def clear(self,mask=None):
        if self.best_cost is not None:
            if mask is None: self.best_cost.fill_(torch.inf)
            else: self.best_cost[mask]=torch.inf
    reset=clear
    def update(self,iteration_state,action_horizon,action_dim,cost_delta_threshold,cost_relative_threshold,convergence_iteration):
        return update_best_solution(iteration_state,action_horizon,action_dim,cost_delta_threshold,cost_relative_threshold,convergence_iteration)
    @staticmethod
    def check_convergence(converged,converged_ratio): return converged.float().mean() >= converged_ratio
