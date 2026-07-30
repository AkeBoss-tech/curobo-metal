"""PyBind CUDA extension boundary."""


def __getattr__(name):
    raise NotImplementedError(
        f"PyBind CUDA extension {name!r} is unavailable on Metal"
    )
