"""SuperMimic: a differentiable world model for humanoid motion tracking."""

# Previously set PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync so the Warp
# MuJoCo CUDA graph shared a mempool with PyTorch. That allocator breaks
# torch.compile(mode="reduce-overhead") CUDA graphs (checkPoolLiveAllocations).
# Leave the default caching allocator; if the sim OOMs, revisit.
