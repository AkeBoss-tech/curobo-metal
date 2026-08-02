from ._launch import unsupported_launch


def launch_kernel(kernel_name, stream, config, kernel, *kernel_args):
    """Reject a raw cuda.core launch without touching stream or pointers.

    The vararg shape matches pinned cuRobo.  Inspecting a tensor's ``data_ptr``
    here would make an unavailable CUDA path look as if it had begun execution.
    """
    del stream, config, kernel, kernel_args
    unsupported_launch(str(kernel_name))


def launch(*args, **kwargs):
    del args, kwargs
    unsupported_launch("launch")
