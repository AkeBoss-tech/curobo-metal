import torch
class L2DistFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,pos,target,target_idx,weight,terminal_dof_weight,
                non_terminal_dof_weight,out_cost_dof,out_gp,use_grad_input):
        if pos.shape[-1] != target.shape[-1]:
            raise ValueError("pos and target must have matching dof")
        if target_idx is None:
            goal = target.expand_as(pos) if target.shape != pos.shape else target
        else:
            idx = target_idx.to(device=target.device, dtype=torch.long).reshape(-1)
            if idx.numel() != pos.shape[0]:
                raise ValueError("target_idx must contain one index per batch")
            goal = target.index_select(0, idx).unsqueeze(-2).expand_as(pos)
        terminal = torch.ones(pos.shape[-1], device=pos.device, dtype=pos.dtype) if terminal_dof_weight is None else terminal_dof_weight.to(pos)
        running = terminal if non_terminal_dof_weight is None else non_terminal_dof_weight.to(pos)
        dof_weight = terminal.expand_as(pos).clone()
        if pos.ndim >= 3 and pos.shape[-2] > 1:
            dof_weight[..., :-1, :] = running
        scale = torch.as_tensor(weight, device=pos.device, dtype=pos.dtype).reshape(-1)[0]
        value=(pos-goal).square()*dof_weight*scale
        ctx.save_for_backward(pos,goal,dof_weight,scale)
        return value
    @staticmethod
    def backward(ctx,grad_out_cost):
        pos,goal,dof_weight,scale=ctx.saved_tensors
        return grad_out_cost*2*(pos-goal)*dof_weight*scale,None,None,None,None,None,None,None,None
def forward_l2_warp(*args,**kwargs):
    raise RuntimeError("forward_l2_warp is a raw Warp kernel; use L2DistFunction or CSpaceDistCost")
__all__=["L2DistFunction","forward_l2_warp"]
