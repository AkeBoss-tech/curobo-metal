from dataclasses import dataclass
import torch
from curobo._src.optim._portable import PortableOptCfg,PortableOptimizer,_objective
@dataclass
class GradientDescentOptCfg(PortableOptCfg):
    solver_type: str="gradient_descent"; solver_name: str="gradient_descent"; inner_iters: int=1
    step_scale: float=0.05; fixed_iters: bool=True; cost_convergence: float=1e-9
    line_search_type: object="GREEDY"
class GradientDescentOpt(PortableOptimizer):
    def optimize(self,seed_action):
        if not self.enabled:return seed_action
        x=seed_action
        objective=_objective(self.rollout_fn)
        for _ in range(self.config.num_iters):
            x=x.requires_grad_(True); cost=objective(x); grad=torch.autograd.grad(cost.sum(),x,create_graph=torch.is_grad_enabled())[0]
            x=x-self.config.step_scale*grad
        return x
class LineSearchGradientDescentOpt(GradientDescentOpt): pass
