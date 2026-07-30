from .portable import StateCSpaceCost as StateCSpaceFunction
def forward_cspace_state_warp(*args,**kwargs):
    raise RuntimeError("raw Warp kernel unavailable; use StateCSpaceCost")
__all__=["StateCSpaceFunction","forward_cspace_state_warp"]
