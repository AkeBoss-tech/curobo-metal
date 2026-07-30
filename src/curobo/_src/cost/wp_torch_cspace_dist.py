import torch
class L2DistFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,pos,target,target_idx,weight,terminal_dof_weight,
                non_terminal_dof_weight,out_cost_dof,out_gp,use_grad_input):
        goal=target if target_idx is None else target.index_select(0,target_idx.reshape(-1)).reshape(pos.shape)
        value=(pos-goal).square()*weight
        ctx.save_for_backward(pos,goal,weight)
        return value
    @staticmethod
    def backward(ctx,grad_out_cost):
        pos,goal,weight=ctx.saved_tensors
        return grad_out_cost*2*(pos-goal)*weight,None,None,None,None,None,None,None,None
def forward_l2_warp(*args,**kwargs):
    raise RuntimeError("forward_l2_warp is a raw Warp kernel; use L2DistFunction or CSpaceDistCost")
__all__=["L2DistFunction","forward_l2_warp"]
