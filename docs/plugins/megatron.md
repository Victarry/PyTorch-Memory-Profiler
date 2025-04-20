# Megatron-LM Integration for Memory Profiler

This integration provides patches to ensure Megatron-LM works correctly with the memory profiler, particularly when analyzing distributed training scenarios.

## Features

The integration addresses three key areas:

1. **Distributed Communication Patches**: Forces the use of specific distributed communication functions:
   - `torch.distributed._all_gather_base`
   - `torch.distributed._reduce_scatter_base`

2. **Multi-Tensor Functions**: Forces the use of local multi-tensor implementations instead of external libraries:
   - `local_multi_tensor_applier`
   - `local_multi_tensor_l2_norm`
   - `local_multi_tensor_scale`

3. **Optimizer Patch**: Replaces `Adam` with `torch.optim.AdamW` in the distributed optimizer

## Usage

The plugin is now registered by default with the `MemoryTracer`. When you create a `MemoryTracer` instance, the Megatron-LM patches will be automatically applied when you enter the context manager:

```python
from memory_profiler import MemoryTracer

# Create memory tracer
tracer = MemoryTracer(device="cuda")

# Use tracer context manager (this will automatically apply the Megatron-LM patches)
with tracer:
    # Your Megatron-LM model code here
    ...
```

### Manual Usage

If you need to apply the patches manually without using the tracer (not recommended), you can import and call the patch function directly:

```python
from memory_profiler.plugins.megatron_core_plugin import apply_megatron_core_patch

# Apply patches
apply_megatron_core_patch()

# Your Megatron-LM code
...
```

## Logging

The plugin logs information about which patches have been applied. You can configure the logging level to get more detailed information:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Technical Details

The patches are applied to the following modules:

1. Distributed functions in:
   - `megatron.core.tensor_parallel.mappings`
   - `megatron.core.distributed.param_and_grad_buffer`
   - `megatron.core.tensor_parallel.layers`
   - `megatron.core.timers`

2. Multi-tensor functions in:
   - `megatron.core.optimizer.clip_grads`

3. Adam optimizer in:
   - `megatron.core.optimizer.distrib_optimizer` 