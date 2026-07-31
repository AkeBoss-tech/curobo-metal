import torch
def get_stomp_cov(horizon,zero_out_boundary=True,stencil_type="3point"):
    if horizon <= 0: raise ValueError("horizon must be positive")
    order=2 if stencil_type=="3point" else 3
    D=torch.zeros(horizon,horizon)
    for i in range(horizon):
        D[i,i]=1
        if i: D[i,i-1]=-1
    cov=torch.linalg.pinv((D.T@D).matrix_power(max(1,order-1)))
    cov=cov/cov.diagonal().max().clamp_min(torch.finfo(cov.dtype).eps)
    if zero_out_boundary and horizon>1: cov[0]=0; cov[:,0]=0; cov[-1]=0; cov[:,-1]=0
    return cov
