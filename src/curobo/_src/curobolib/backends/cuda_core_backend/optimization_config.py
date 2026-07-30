from ._launch import LaunchConfig


class OptimizationKernelCfg:
    def get_kernel_files(self, kernel_type: str):
        return {"line_search": ["line_search/line_search_kernel.cuh"], "lbfgs": ["lbfgs/lbfgs_step_kernel.cuh"]}.get(kernel_type, [])
    def get_include_dirs(self): return []


class LineSearchLaunchCfg:
    @staticmethod
    def calculate_config(opt_dim, batchsize): return LaunchConfig(batchsize, opt_dim, 0)


class LBFGSLaunchCfg:
    @staticmethod
    def calculate_config(batch_size, v_dim, history_m, use_shared_buffers):
        basic = history_m * 4
        requested = (((2 * v_dim) + 2) * history_m + 33) * 4
        actual = bool(use_shared_buffers and requested <= 65536)
        shmem = requested if use_shared_buffers else basic
        return LaunchConfig(batch_size, v_dim, shmem), actual, requested if actual else 48000
