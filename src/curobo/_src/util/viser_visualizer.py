"""Optional Viser visualization adapter."""
class ViserVisualizer:
    def __init__(self,*args,**kwargs):
        try:import viser
        except ImportError as e:raise ImportError("Viser not installed. Install with: pip install viser") from e
        raise NotImplementedError("the cuRobo Viser robot adapter is unavailable on the portable backend")
