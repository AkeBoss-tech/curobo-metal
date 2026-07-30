from dataclasses import dataclass


@dataclass(frozen=True)
class LaunchConfig:
    grid: object
    block: object
    shmem_size: int = 0


def unsupported_launch(name: str):
    raise NotImplementedError(
        f"{name} is a raw CUDA kernel launch and cannot run on Metal; use the "
        "corresponding curobo-metal production operator"
    )
