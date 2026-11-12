"""CUDA Graph replay with refreshed inputs and stable model/optimizer storage."""

import torch


def _optimizers(optimizer):
    return getattr(optimizer, "optimizers", [optimizer])


def _tensor_state(modules, optimizers):
    values = []
    for module in modules:
        values.extend(module.parameters())
        values.extend(module.buffers())
    for group in optimizers:
        for optimizer in _optimizers(group):
            for state in optimizer.state.values():
                values.extend(x for x in state.values() if isinstance(x, torch.Tensor))
    return values


class CapturedUpdate:
    """Warm up real updates, capture once, then refresh inputs and replay.

    Separate graph pools keep WM and policy outputs independent. Capture itself
    records operations; the first replay is the one update for that call.
    """

    def __init__(self, fn, state, enabled, warmup=3):
        self.fn, self.state, self.enabled = fn, state, enabled
        self.warmup = warmup
        self.inputs = self.graph = self.outputs = None
        self.identity = None
        self.stream = torch.cuda.Stream() if enabled else None

    def __call__(self, *inputs, allow_capture=True):
        if not self.enabled or not allow_capture:
            return self.fn(*inputs)
        if self.inputs is None:
            self.inputs = tuple(x.clone() for x in inputs)
        else:
            for destination, source in zip(self.inputs, inputs, strict=True):
                if destination.shape != source.shape or destination.dtype != source.dtype:
                    raise ValueError("Captured update input shape/dtype changed")
                destination.copy_(source)
        if self.warmup:
            self.warmup -= 1
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                result = self.fn(*self.inputs)
            torch.cuda.current_stream().wait_stream(self.stream)
            return result
        if self.graph is None:
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=self.stream):
                self.outputs = self.fn(*self.inputs)
            self.identity = tuple(t.data_ptr() for t in self.state())
        elif self.identity != tuple(t.data_ptr() for t in self.state()):
            raise RuntimeError("Captured parameter, normalizer or optimizer storage moved")
        self.graph.replay()
        return self.outputs


def optimize_capturable(loss, module, optimizer, max_norm, clip_value=0.0):
    # Device assertions preserve fail-fast checks without Tensor -> bool syncs.
    torch._assert_async(torch.isfinite(loss), "Non-finite learner loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm, error_if_nonfinite=False)
    torch._assert_async(torch.isfinite(norm), "Non-finite learner gradient norm")
    if clip_value > 0:
        torch.nn.utils.clip_grad_value_(module.parameters(), clip_value)
    optimizer.step()
    return norm.detach()
