# Launch post drafts

The IK fast path has passed the local release gates in a clean `1.0.1` wheel.
These drafts still treat it as a release candidate until the tag is published
and a fresh install from PyPI is verified. Replace `[1.0.1 URL]` only after that
patch release exists.

## Recommended launch order

1. Publish a GitHub release with the verification matrix and reproducible
   benchmark commands.
2. Post a concise technical announcement to ROS Discourse and r/robotics to
   recruit Apple Silicon users with real workloads.
3. Use a `Show HN` post after the README, install path, and patch wheel have had
   at least one independent clean-environment confirmation.
4. Share the visual and the key trade-off on LinkedIn or Mastodon, linking back
   to the technical release rather than presenting it as performance parity.

## GitHub release

### cuRobo Metal 1.0.1 — cuRobo-compatible planning on Apple Silicon

`curobo-metal` provides the portable public Python surface of pinned cuRoboV2
on CPU and Apple Metal. Install it in a clean environment with:

```bash
pip install curobo-metal
```

It exposes the `curobo` namespace, so NVIDIA cuRobo must not be installed in the
same environment. The compatibility gate matches 24 public modules, 122
exports, and 10 comparable callable signatures at upstream commit
`8e734f3ced1df898990bcd92de40abce475907db`.

This patch short-circuits IK refinement when the LM seed already meets the
requested tolerances. On the development M4, a repeated moved-target Franka IK
check improved from 10/20 successes at about 315 ms per successful solve in
1.0.0 to 20/20 at about 17.9 ms median in the patch candidate.

This is not CUDA performance parity. In matched warm public-API tests, an RTX
3090 ran batch-1,024 FK in 0.372 ms versus 3.79 ms on M4, and the same focused
moved-target IK check in 5.87 ms versus 17.9 ms after the patch. Joint-space
planning was much closer: 89.9 ms on RTX 3090 versus 110 ms on M4. See the
README for cold-start numbers, the broader IK release gate, methodology,
limitations, and the generated Franka planning demo.

Feedback with real robots, scenes, and minimal reproductions is especially
welcome: [1.0.1 URL]

## ROS Discourse / r/robotics

### cuRobo-compatible motion planning now runs natively on Apple Silicon

I have been building `curobo-metal`, an independent CPU/MPS implementation of
the portable cuRoboV2 Python API. The goal is to let robotics developers run
FK, collision queries, IK, and trajectory planning locally on Apple Silicon
without rewriting portable cuRobo code.

The current compatibility gate matches 24 public modules and 122 exports
against a pinned upstream revision. A motion-planning demo rendered with the
bundled Franka meshes, and the code that generated it, are in the repository.

The honest performance picture: an M4 is useful, but it does not beat CUDA for
warm IK. After the next patch, repeated moved-target IK is about 17.9 ms on the
M4 versus 5.87 ms on an RTX 3090; joint-space planning is much closer at roughly
110 ms versus 89.9 ms. Cold starts can favor the Mac. The README contains the
protocol and the exact compatibility boundary.

I would value reports from people using non-Franka robots, collision-heavy
scenes, or differentiable planning. What breaks, and which workload should be
optimized next?

Repository: https://github.com/AkeBoss-tech/curobo-metal

## Show HN

### Show HN: cuRobo Metal – robot motion planning on Apple Silicon

I built an Apple Silicon implementation of the portable Python API in a pinned
cuRoboV2 revision. It installs as `curobo-metal` but exposes the familiar
`curobo` namespace, with CPU/MPS FK, collision, IK, trajectory optimization,
mapping, and deterministic geometric planning.

I tested it as an external user from a clean environment and compared the same
public calls with cuRobo on an RTX 3090. CUDA is still much faster for warm FK
and IK; joint-space planning was closer, and the Mac had lower cold-start cost
in the tested cases. I have published the unflattering numbers too, plus the
scripts and compatibility boundary, because I want this to be useful rather
than a parity claim.

The README includes a minimal example and a generated Franka planning GIF. I am
particularly interested in feedback on where native Mac robotics tooling is
actually useful and which solver hot path is worth fusing next.

https://github.com/AkeBoss-tech/curobo-metal

## LinkedIn / Mastodon

Robot motion planning on a Mac, through the public cuRobo API.

I have released `curobo-metal`, an independent Apple Silicon CPU/MPS
implementation targeting a pinned cuRoboV2 contract. It supports portable FK,
collision, IK, and trajectory-planning workflows and includes a reproducible
Franka visual demo.

The benchmark story is candid: an RTX 3090 remains much faster for warm IK,
while the tested joint-space plan was relatively close and Apple cold-start
latency was lower. The project is for local development, research, education,
and CI—not a claim that Metal has reached CUDA parity.

If you work in robotics on Apple Silicon, I would love a real workload and a
bug report. https://github.com/AkeBoss-tech/curobo-metal
