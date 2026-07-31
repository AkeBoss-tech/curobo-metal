from dataclasses import dataclass


@dataclass
class ProjectiveTextureProjectorCfg:
    num_cameras: int = 1
    image_height: int | None = None
    image_width: int | None = None


class ProjectiveTextureProjector:
    def __init__(self, cfg=None, **kwargs):
        self.cfg = cfg or ProjectiveTextureProjectorCfg(**kwargs)

    def project(self, *args, **kwargs):
        raise NotImplementedError(
            "projective texture fusion is a Warp/CUDA-only optional mapper feature"
        )

    __call__ = project
