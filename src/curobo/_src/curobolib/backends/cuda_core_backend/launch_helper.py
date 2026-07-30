from ._launch import unsupported_launch


def launch_kernel(kernel_name, stream, config, kernel, kernel_args):
    unsupported_launch(kernel_name)


def launch(*args, **kwargs):
    unsupported_launch("launch")
