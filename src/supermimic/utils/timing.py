"""Per-iteration phase spans, including asynchronous CUDA work and launch gaps."""

import time

import torch


class IterationTimer:
    def __init__(self, device):
        self.device = torch.device(device)
        self.events = (
            [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            if self.device.type == "cuda"
            else None
        )

    def start(self):
        self.started = time.perf_counter()
        if self.events:
            self.events[0].record(torch.cuda.current_stream(self.device))

    def collected(self):
        self.collect_done = time.perf_counter()
        if self.events:
            self.events[1].record(torch.cuda.current_stream(self.device))

    def finish(self):
        if self.events:
            self.events[2].record(torch.cuda.current_stream(self.device))
            # One synchronization per iteration; otherwise graph enqueue time
            # would be incorrectly reported as completed learner execution.
            self.events[2].synchronize()
            collect = self.events[0].elapsed_time(self.events[1]) / 1000
            train = self.events[1].elapsed_time(self.events[2]) / 1000
        else:
            collect = self.collect_done - self.started
            train = time.perf_counter() - self.collect_done
        return {
            "collect_time": collect,
            "train_time": train,
            "compute_time": time.perf_counter() - self.started,
        }
