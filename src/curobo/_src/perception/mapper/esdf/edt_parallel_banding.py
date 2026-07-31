from .edt_jump_flooding import JumpFloodingEDT


class ParallelBandingEDT(JumpFloodingEDT):
    """Portable exact EDT; PBA's raw CUDA scheduling is intentionally absent."""
