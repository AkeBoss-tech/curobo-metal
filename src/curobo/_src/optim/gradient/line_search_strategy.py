from enum import Enum
import torch
from .line_search_state import LineSearchState
from .line_search_result import LineSearchResult
class LineSearchType(Enum): GREEDY="GREEDY"; ARMIJO="ARMIJO"; WOLFE="WOLFE"; STRONG_WOLFE="STRONG_WOLFE"; APPROX_WOLFE="APPROX_WOLFE"; APPROX_STRONG_WOLFE="APPROX_STRONG_WOLFE"
class LineSearchStrategy:
    @staticmethod
    def jit_get_x_set(step_vec,x,line_search_scales): return x.unsqueeze(-2)+line_search_scales.reshape((1,)*x.ndim+(-1,1))*step_vec.unsqueeze(-2)
    @staticmethod
    def scale_action(dx,action_step_max,step_scale,fix_terminal_action,action_horizon):
        out=torch.clamp(dx,min=-action_step_max,max=action_step_max)*step_scale
        if fix_terminal_action and action_horizon>0: out[...,action_horizon-1,:]=0
        return out
    def search(self,iteration_state,context):
        scales=context.line_search_scale
        xs=iteration_state.action.unsqueeze(-3)+scales.reshape((-1,)+(1,)*(iteration_state.action.ndim-1))*iteration_state.step_direction.unsqueeze(-3)
        cost,grad=context.compute_costs_and_gradients(xs)
        idx=cost.argmin(dim=-1); gather=idx.reshape(idx.shape+(1,)*(xs.ndim-idx.ndim)).expand(idx.shape+xs.shape[idx.ndim:])
        action=torch.gather(xs,idx.ndim,gather).squeeze(idx.ndim)
        selected=LineSearchState(action,torch.gather(cost,-1,idx.unsqueeze(-1)).squeeze(-1),torch.gather(grad,idx.ndim,gather).squeeze(idx.ndim),idx)
        return LineSearchResult(selected,selected)
class GreedyLineSearchStrategy(LineSearchStrategy): pass
class ArmijoLineSearchStrategy(LineSearchStrategy): pass
class BaseWolfeLineSearchStrategy(LineSearchStrategy): pass
class WolfeLineSearchStrategy(BaseWolfeLineSearchStrategy): pass
class StrongWolfeLineSearchStrategy(BaseWolfeLineSearchStrategy): pass
class ApproxWolfeLineSearchStrategy(BaseWolfeLineSearchStrategy): pass
class ApproxStrongWolfeLineSearchStrategy(BaseWolfeLineSearchStrategy): pass
class LineSearchStrategyFactory:
    _map={t:GreedyLineSearchStrategy for t in LineSearchType}; _map[LineSearchType.ARMIJO]=ArmijoLineSearchStrategy
    @classmethod
    def get_strategy(cls,strategy_type): return cls._map[strategy_type]()
    @classmethod
    def register_strategy(cls,strategy_type,strategy): cls._map[strategy_type]=strategy if isinstance(strategy,type) else type(strategy)
