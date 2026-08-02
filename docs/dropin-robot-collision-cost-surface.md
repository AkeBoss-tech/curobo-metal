# Portable robot, collision, and cost surface

This compatibility wave fills callable seams that appear in ordinary cuRobo V2
robot-planning programs while routing all execution through the existing
portable PyTorch CPU/MPS backends.

`Kinematics.get_full_js`, `get_mimic_js`, and equivalent-configuration updates
are available for already-compiled portable trees.  Changing tree topology or
recovering a nonzero locked revolute transform still requires constructing a
new model.  Mesh construction remains the responsibility of the parser; it is
not silently emulated by the FK backend.

Configuration-space, scene-collision, and self-collision costs now expose the
batch lifecycle and validation helpers expected by cost managers.  The
quaternion helpers use normal differentiable PyTorch operations.  They are
portable tensor utilities, not Warp kernel/stream ABI replacements.

Raw CUDA Graph, Warp launch, pointer-buffer, Isaac, and NVIDIA-specific
numerical-parity claims remain explicitly out of scope.
