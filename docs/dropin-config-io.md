# Portable configuration I/O

`curobo.config_io` and `curobo._src.util.config_io` retain the pinned V2 YAML,
path, merge, file-copy, and platform helper surface.  A string or `Path` loads
YAML; a parsed mapping or typed record passes through unchanged, matching the
normal cuRobo configuration handoff.

The portable extension exposes `resolve_device_cfg` and `resolve_dataclass`.
They make serializable configuration loading explicit rather than relying on
CUDA-era implicit global device state:

```python
cfg = resolve_dataclass(
    PlannerCfg,
    {"device_cfg": {"device": "mps", "dtype": "float32"}},
)
```

Supported device descriptions are CPU and MPS.  CUDA requests fail immediately
with `NotImplementedError`; no CPU fallback is selected.  Dataclass mapping
keys are checked, so unknown keys raise `ConfigIOError` rather than being
silently ignored.  `write_yaml` serializes ordinary mappings plus portable
dataclasses, `DeviceCfg`, and `Path` values to standard YAML.

XRDF remains YAML-shaped and can be loaded by the robot/XRDF compatibility
layer.  USD/USDA/USDC/USDZ and Isaac configuration paths fail explicitly: they
need OpenUSD/Isaac scene semantics which are outside this PyTorch Metal port.
