from ._launch import LaunchConfig
from .kernel_config import CudaCoreKernelCfg
from .util import ceil_div


class TrajectoryKernelCfg(CudaCoreKernelCfg):
    def __init__(self): super().__init__("trajectory")
    def get_kernel_files(self, kernel_type: str):
        return {
            "bspline_forward": ["bspline/bspline_kernel.cuh"],
            "bspline_backward": ["bspline/bspline_kernel.cuh"],
            "bspline_single_dt": ["bspline/bspline_kernel.cuh"],
            "differentiation_forward": ["legacy/differentiation_position_kernel.cuh"],
            "differentiation_backward": ["legacy/differentiation_position_kernel.cuh"],
            "integration": ["legacy/integration_acceleration_kernel.cuh"],
        }.get(kernel_type, [])
    def get_include_dirs(self): return self.get_base_include_dirs() + [self.kernel_dir]


class BSplineBackwardLayout:
    def __init__(self):
        self.interpolation_steps = self.knots_per_warp = self.warps_for_n_knots = 0
        self.threads_for_n_knots = self.padded_horizon = self.n_knots = 0
        self.padded_n_knots = self.horizon = self.dof = 0


def get_spline_support_size(degree): return degree + 1
def get_total_knots(n_knots, degree): return n_knots + get_spline_support_size(degree)


def compute_bspline_backward_layout(horizon, dof, n_knots, bspline_degree):
    layout = BSplineBackwardLayout()
    layout.padded_n_knots = get_total_knots(n_knots, bspline_degree)
    layout.interpolation_steps = horizon // layout.padded_n_knots
    if layout.interpolation_steps <= 0:
        raise ValueError("horizon must cover the padded knot support")
    layout.knots_per_warp = 32 // layout.interpolation_steps
    if layout.knots_per_warp <= 0:
        raise RuntimeError("interpolation_steps greater than 32 is unsupported")
    layout.warps_for_n_knots = ceil_div(n_knots, layout.knots_per_warp)
    layout.threads_for_n_knots = layout.warps_for_n_knots * 32
    layout.padded_horizon, layout.n_knots = horizon + 1, n_knots
    layout.horizon, layout.dof = horizon, dof
    return layout


class BSplineLaunchCfg:
    @staticmethod
    def _linear(size, limit):
        threads = min(size, limit)
        return LaunchConfig(ceil_div(size, threads), threads, 0)
    @staticmethod
    def calculate_forward_config(batch_size, dof, horizon): return BSplineLaunchCfg._linear(batch_size*dof*horizon, 128)
    @staticmethod
    def calculate_backward_config(batch_size, dof, n_knots, horizon, bspline_degree):
        layout = compute_bspline_backward_layout(horizon, dof, n_knots, bspline_degree)
        return BSplineLaunchCfg._linear(batch_size*dof*layout.threads_for_n_knots, 128)
    @staticmethod
    def calculate_single_dt_config(batch_size, dof, max_out_tsteps): return BSplineLaunchCfg._linear(batch_size*dof*max_out_tsteps, 256)


class LegacyTrajectoryLaunchCfg:
    @staticmethod
    def calculate_differentiation_forward_config(batch_size, dof, horizon): return BSplineLaunchCfg._linear(batch_size*dof*horizon, 128)
    @staticmethod
    def calculate_differentiation_backward_config(batch_size, dof, horizon): return BSplineLaunchCfg._linear(batch_size*dof*(horizon-4), 128)
    @staticmethod
    def calculate_integration_config(batch_size, dof): return BSplineLaunchCfg._linear(batch_size*dof, 512)
