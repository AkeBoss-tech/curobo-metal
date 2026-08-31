from ._launch import unsupported_launch
import curobo._src.runtime as runtime
from curobo._src.util.logging import log_and_raise


class _CudaRuntimeUnavailable:
    """Fail explicitly when code crosses the raw cuda.bindings boundary."""

    def __getattr__(self, name):
        unsupported_launch(f"cuda.bindings.runtime.{name}")


# Keep the pinned import names available without importing CUDA-only wheels.
cudart = _CudaRuntimeUnavailable()


def launch_kernel(kernel_name, stream, config, kernel, *kernel_args):
    """Reject a raw cuda.core launch without touching stream or pointers.

    The vararg shape matches pinned cuRobo.  Inspecting a tensor's ``data_ptr``
    here would make an unavailable CUDA path look as if it had begun execution.
    """
    del stream, config, kernel, kernel_args
    unsupported_launch(str(kernel_name))


def _launch_portable(*args, **kwargs):
    del args, kwargs
    unsupported_launch("launch")


# Upstream re-exports ``cuda.core.launch``.  The portable equivalent is a
# callable raw-CUDA boundary, retained as an assignment so its declaration
# category stays compatible without pretending CUDA launches are supported.
launch = _launch_portable
