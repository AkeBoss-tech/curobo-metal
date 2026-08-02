# Public configuration, content, and runtime helpers

The public `curobo.config_io`, `curobo.content`, `curobo.scene`,
`curobo.types`, `curobo.logging`, `curobo.profiling`, `curobo.sphere_fit`,
and `curobo.viewer` imports match the pinned cuRoboV2 layout.  They are safe
to import in a CPU-only or Apple Metal Python process: importing them does not
initialize CUDA, Warp, Isaac Sim, OpenUSD, or Viser.

`curobo.content` locates files inside the installed distribution, so the same
helpers work from a source checkout and an installed wheel.  This port bundles
the Franka robot and primitive-scene assets used by the supported planning
examples; callers should supply their own assets for the broader upstream
content catalog.

`CudaEventTimer` retains its cuRobo name and `start().stop()` protocol.  It
returns seconds.  CUDA uses events when available; MPS synchronizes at the
measurement boundaries and reports monotonic end-to-end time.  Setting
`curobo._src.runtime.cuda_event_timers = False` makes it return `0.0`, as in
the pinned runtime contract.  It also supports `with CudaEventTimer() as t:`
as a non-breaking convenience.

`curobo.viewer.UsdWriter` and `curobo.viewer.ViserVisualizer` are lazy optional
entry points.  They raise a direct install/unsupported error when `usd-core` or
`viser` is absent; they do not silently substitute a viewer.  Isaac-specific
scene conversion and the full Viser robot adapter remain outside the portable
backend.

Sphere fitting is a deterministic PyTorch CPU/MPS approximation that preserves
the public `SphereFitType`, result, metrics, count-estimation, and fitting
interfaces.  It is not a numerical claim of the pinned Warp sphere-fitting
kernel or mesh-BVH implementation.
