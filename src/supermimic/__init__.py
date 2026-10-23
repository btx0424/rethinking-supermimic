"""SuperMimic: a differentiable world model for humanoid motion tracking."""

import os
import sys

# The MuJoCo step is a Warp CUDA graph on the device mempool. PyTorch's default
# caching allocator takes memory outside that pool, and the graph launch then
# fails with "out of memory". cudaMallocAsync uses the same pool. This must be
# set before the first torch import; setting it afterwards makes torch abort.
if "torch" not in sys.modules:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
