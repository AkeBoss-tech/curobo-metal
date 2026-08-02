from dataclasses import dataclass


@dataclass(frozen=True)
class LaunchConfig:
    grid: object
    block: object
    shmem_size: int = 0


class RawCudaKernelUnavailableError(NotImplementedError):
    """A CUDA ABI entry point with no semantically safe Metal substitute."""


def unsupported_launch(name: str, *, alternative: str | None = None) -> None:
    detail = alternative or "the corresponding curobo-metal production operator"
    raise RawCudaKernelUnavailableError(
        f"{name} is a raw CUDA kernel launch and cannot run on Metal; use {detail}"
    )
