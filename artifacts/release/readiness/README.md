# Alpha validation checkpoint

The final `0.1.0a1` candidate passes the bounded application compatibility gate.
This is not a stable 1.0 or whole-ecosystem drop-in certification.

- Eight byte-identical portable V2 applications pass on Robo (RTX 3090) and Apple MPS.
- The final CUDA/Metal comparison has zero differences at the manifest tolerances.
- The candidate wheel passes all eight applications on Python 3.10, 3.11, 3.12 and 3.13.
- The local suite passes 1,390 tests with MPS fallback disabled.
- The final targeted upstream IK and humanoid retargeting run passes all 45 cases.
- Public API facade, wheel/sdist metadata, archive content, dependency and sdist smoke checks pass.

`report.json` identifies the exact archives and report hashes. `cuda-metal-comparison.json`
attests the final paired application reports. `release-application-bundle` contains the
source-identical replay programs and final Metal report. Historical reports are retained;
only the paths named in `report.json` identify the final candidate's application evidence.
The earlier full upstream census and its targeted corrections are identified separately.

CUDA validation led to fixes for full IK joint-state results, active-joint retargeter
velocities, and robot configurations with null locked joints. Earlier changes fixed
automatic MPS device selection, collision constraints and zero-attempt planning.

Remaining publication work: review/commit the worktree, create the release tag, and
configure or verify PyPI trusted publishing. Nothing has been uploaded. The wider stable
contract still has private dependency re-export gaps and unvalidated application/performance
coverage; the eight programs do not establish all API, mapping, or real-time parity.
