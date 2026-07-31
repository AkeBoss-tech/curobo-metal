import math
import numpy as np
import torch
def rotation_error_quaternion(q_des,q):
    return np.minimum(torch.norm(q_des+q).detach().cpu().numpy(),torch.norm(q_des-q).detach().cpu().numpy())/math.sqrt(2)
def rotation_error_matrix(r_des,r):
    return .5*torch.square(r-r_des).sum(dim=(-2,-1))
__all__=["rotation_error_quaternion","rotation_error_matrix"]
