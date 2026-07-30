from .portable import PositionCSpaceCost as PositionCSpaceFunction
def forward_cspace_position_warp(*args,**kwargs):
    raise RuntimeError("raw Warp kernel unavailable; use PositionCSpaceCost")
__all__=["PositionCSpaceFunction","forward_cspace_position_warp"]
