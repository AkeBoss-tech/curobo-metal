"""Standalone synchronized graph-planning latency and outcome benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from curobo_metal.ops.graph_planning import GraphPlanningProblem, plan_graph


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--fixture", default="two_link_detour.json")
    args = parser.parse_args()
    device, dtype = torch.device(args.device), torch.float32
    root = Path(__file__).parents[2]
    case = json.loads((root / "tests/fixtures/graph_planning" / args.fixture).read_text())
    inputs, options = case["inputs"], case["options"]
    tensor = lambda value: torch.tensor(value, device=device, dtype=dtype)
    lower = tensor(inputs["validity"]["lower"]).reshape(-1, len(inputs["lower"]))
    upper = tensor(inputs["validity"]["upper"]).reshape_as(lower)

    def validity(q: torch.Tensor) -> torch.Tensor:
        if lower.shape[0] == 0:
            return torch.ones(q.shape[0], dtype=torch.bool, device=device)
        return ~(((q[:, None] >= lower) & (q[:, None] <= upper)).all(-1)).any(-1)

    problem = GraphPlanningProblem(
        tensor(inputs["starts"]), tensor(inputs["goals"]), tensor(inputs["lower"]),
        tensor(inputs["upper"]), validity=validity, **options,
    )
    start = time.perf_counter()
    result = plan_graph(problem)
    synchronize(device)
    warmup_ms = 1e3 * (time.perf_counter() - start)
    samples = []
    for _ in range(args.repeats):
        synchronize(device)
        start = time.perf_counter()
        result = plan_graph(problem)
        synchronize(device)
        samples.append(1e3 * (time.perf_counter() - start))
    metric = result.metrics[0]
    print(json.dumps({
        "device": args.device, "dtype": str(dtype), "fixture": args.fixture,
        "compile_warmup_ms": warmup_ms,
        "latency_ms": {"median": torch.tensor(samples).median().item(), "samples": samples},
        "success": result.success.tolist(), "status": list(result.status),
        "path_cost": metric.path_cost, "samples_valid": metric.samples_valid,
        "edge_checks": metric.edge_checks, "validity_queries": metric.validity_queries,
    }, indent=2))


if __name__ == "__main__":
    main()
