# Depth-filter compatibility

`curobo._src.perception.filter_depth.FilterDepth` is implemented with regular
PyTorch tensors on CPU and float32 MPS. It retains the V2 parameter object,
constructor and `from_config` spellings, reusable `._depth_out` and
`._valid_mask_out` buffers, dynamic-shape fallback buffers, caller-owned
output buffers, range/finite-value rejection, flying-pixel configuration,
bilateral filtering, and non-allocating `update_config` lifecycle.

The source-shaped contract is a float32 `(B, H, W)` tensor and a `(B, H, W)`
bool validity mask. The portable facade also accepts the legacy unbatched
`(H, W)` input form when no caller-owned output buffer is requested; it returns
an unbatched pair. New drop-in code should use the V2 batched form.

For threshold values in the documented `0..1` range, flying-pixel tolerance
uses the V2 logarithmic relative-depth mapping. `update_config` treats zero
as the upstream disable request. Values larger than one are retained as a
portable legacy absolute-tolerance extension for pre-existing Metal callers;
they are not a claim of CUDA/Warp numerical parity.

The CUDA/Warp fused and separable kernel entry points, CUDA graph state, raw
Warp output buffers, and kernel ABI are deliberately not exposed. The MPS
implementation uses a direct 2-D bilateral operation for all kernel sizes,
which preserves the documented filtering semantics while not promising the
source's large-kernel separable approximation or its timing.
