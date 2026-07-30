from __future__ import annotations

from pathlib import Path
import torch


class GraphExecutor:
    """Shape-stable direct executor used in place of CUDA graph capture."""

    def __init__(self, capture_fn, device, use_cuda_graph=None, clone_outputs=True, **capture_fn_kwargs):
        self._capture_fn = capture_fn
        self._device = torch.device(device)
        self._use_cuda_graph = bool(use_cuda_graph)
        self._clone_outputs = clone_outputs
        self._capture_fn_kwargs = capture_fn_kwargs
        self._graph_input = None
        self._graph_output = None
        self._graph = None

    def __call__(self, *inputs, clone_outputs=None):
        if self._graph_input is None:
            self._initialize(inputs)
        elif len(inputs) != len(self._graph_input) or any(
            isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape != b.shape
            for a, b in zip(inputs, self._graph_input)
        ):
            self.reset()
            self._initialize(inputs)
        else:
            self._graph_input = tuple(x.clone() if isinstance(x, torch.Tensor) else x for x in inputs)
        output = self._execute_direct()
        clone = self._clone_outputs if clone_outputs is None else clone_outputs
        if clone:
            output = tuple(x.clone() if hasattr(x, "clone") else x for x in output)
        return output[0] if len(output) == 1 else output

    def warmup(self, *sample_inputs):
        if self._graph_input is None: self._initialize(sample_inputs)
        return self

    def _initialize(self, inputs): self._initialize_direct(inputs)
    def _initialize_cuda_graph(self, inputs):
        raise NotImplementedError("raw CUDA graph capture is unavailable on Metal")
    def _initialize_direct(self, inputs):
        self._graph_input = tuple(x.clone() if isinstance(x, torch.Tensor) else x for x in inputs)
        self._graph_output = self._execute_direct()
    def _execute_direct(self):
        value = self._capture_fn(*self._graph_input, **self._capture_fn_kwargs)
        return value if isinstance(value, tuple) else (value,)
    def reset(self):
        self._graph_input = self._graph_output = self._graph = None
    def debug_dump(self, file_path):
        Path(file_path).write_text("Portable GraphExecutor: no CUDA graph exists.\\n")
    @property
    def is_initialized(self): return self._graph_input is not None


def create_graph_executor(capture_fn, device, use_cuda_graph=None, clone_outputs=False, **capture_fn_kwargs):
    return GraphExecutor(capture_fn, device, use_cuda_graph, clone_outputs, **capture_fn_kwargs)
