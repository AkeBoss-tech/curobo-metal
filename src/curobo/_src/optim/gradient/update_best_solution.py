import torch
def update_best_solution(iteration_state, action_horizon, action_dim, cost_delta_threshold, cost_relative_threshold, convergence_iteration):
    del action_horizon, action_dim
    cost=iteration_state.cost
    if cost is None: return iteration_state
    if iteration_state.best_cost is None:
        iteration_state.best_cost=cost.clone(); iteration_state.best_action=iteration_state.action.clone()
        iteration_state.best_iteration=torch.zeros_like(cost,dtype=torch.long)
    better=cost < iteration_state.best_cost
    iteration_state.best_cost=torch.where(better,cost,iteration_state.best_cost)
    iteration_state.best_action=torch.where(better.reshape(better.shape+(1,)*(iteration_state.action.ndim-better.ndim)),iteration_state.action,iteration_state.best_action)
    if iteration_state.current_iteration is not None:
        iteration_state.best_iteration=torch.where(better,iteration_state.current_iteration,iteration_state.best_iteration)
    abs_ok=(cost-iteration_state.best_cost).abs() <= cost_delta_threshold
    rel_ok=(cost-iteration_state.best_cost).abs() <= cost_relative_threshold*iteration_state.best_cost.abs().clamp_min(1)
    iteration_state.converged=(abs_ok|rel_ok)&(iteration_state.best_iteration>=convergence_iteration)
    return iteration_state
