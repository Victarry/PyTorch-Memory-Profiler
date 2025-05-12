# PyTorch Memory Tracing

A utility library designed to estimate and analyze the memory footprint of PyTorch models, particularly large models, often without requiring actual GPU resources or full datasets. This tool leverages PyTorch's `FakeTensorMode` and `TorchDispatchMode` capabilities to simulate model execution and track estimated memory allocation.

## Features

*   **Memory Estimation:** Estimate peak and current memory usage for model training steps (forward, backward, optimizer) using fake tensors on CPU or GPU.
*   **Module-Level Tracking:** Pinpoint memory usage to specific modules within your model hierarchy using PyTorch hooks.
*   **Training Phase Analysis:** Analyze memory allocation patterns across different training phases (e.g., "forward", "backward", "optimizer") using the `track_phase` context manager.
*   **Fake Tensor Simulation:** Utilizes `FakeTensorMode` to represent large tensors without allocating actual memory, enabling analysis of models that might exceed available hardware memory.
*   **Extensible Plugin System:** Handles compatibility with external libraries via plugins that patch relevant functions to work correctly with fake tensors. Default plugins support:
    *   `torch.distributed`
    *   `megatron-core` (specifically P2P communication for pipeline parallelism)
    *   `transformer-engine`
*   **Detailed Reporting:** Generate reports on peak/current memory, memory usage per module, memory usage per phase, and tensor creation statistics.
*   **Structured Logging System:** Provides a flexible logging system that allows configuration of log levels, console and file output, and rich formatting.

## Requirements

*   PyTorch >= 1.12.0 (due to reliance on `FakeTensorMode` and `TorchDispatchMode`)
*   Optional:
    *   `megatron-core`: For tracing memory in models using Megatron's pipeline parallelism.
    *   `transformer-engine`: For tracing memory in models using Transformer Engine layers.
    *   `rich`: For enhanced log formatting and visual output.

## Installation

```bash
# Basic installation
pip install .

# Installation with optional dependencies for Megatron-LM and Transformer Engine
# Ensure megatron-core and transformer-engine are installed separately first
# pip install .[megatron,te] # (Assuming setup.py extras_require is configured)
```
*(Note: Requires a `setup.py` or `pyproject.toml` with potential `extras_require` for optional dependencies. Adjust installation commands based on your project setup.)*

## Quick Start

Here's a basic example demonstrating how to use `MemoryTracer`:

```python
import torch
import torch.nn as nn
from memory_profiler import MemoryTracer

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        return self.linear(x)

# Configuration
hidden_size = 1024
vocab_size = 1000
batch_size = 4
seq_length = 128
device = 'cuda' # Or 'cpu'
dtype = torch.float16

# Initialize the tracer
# The tracer automatically uses FakeTensorMode and MemoryDispatchMode
estimator = MemoryTracer(device=device)

# Use the estimator context
with estimator:
    # Create model on the target device (will be fake if device='meta' or during tracing)
    model = SimpleModel(hidden_size, vocab_size).to(device=device, dtype=dtype)

    # Register hooks to track memory within the model's modules
    hook_handles = estimator.memory_dispatch_mode.register_hooks_to_module(model)

    # Create optimizer (parameters will be fake during tracing)
    optimizer = torch.optim.Adam(model.parameters())

    # Create fake input tensor
    fake_input = estimator.create_fake_tensor(
        batch_size, seq_length,
        dtype=torch.long, # Adjust dtype as needed
        device=device,
    )

    # --- Simulate Training Iteration with Phase Tracking ---
    optimizer.zero_grad()

    with estimator.track_phase("forward"):
        output = model(fake_input)
        print(f"Estimated memory after forward: {estimator.get_current_memory_allocated()[0] / (1024 ** 2):.2f} MB")
        # Fake loss calculation
        loss = output.sum()

    with estimator.track_phase("backward"):
        loss.backward()
        print(f"Estimated memory after backward: {estimator.get_current_memory_allocated()[0] / (1024 ** 2):.2f} MB")

    with estimator.track_phase("optimizer"):
        optimizer.step()
        print(f"Estimated memory after optimizer step: {estimator.get_current_memory_allocated()[0] / (1024 ** 2):.2f} MB")

    # --- End Simulation ---

    # Clean up hooks
    estimator.memory_dispatch_mode.remove_hooks(hook_handles)

# Print summary reports
estimator.print_memory_stats(detailed=True) # Set detailed=True for module/op level info

print(f"\nPeak estimated memory across all phases: {estimator.get_max_memory_allocated()[0] / (1024 ** 2):.2f} MB")
```
*(See `examples/simple/single_gpu.py` for a runnable version without phase tracking.)*

## Logging System

The pytorch-memory-tracing library includes a structured logging system that replaces the previous print statements with configurable logging. This allows for better control over log output and integration with existing logging systems.

### Basic Usage

```python
import logging
from memory_profiler import MemoryTracer, configure_logging, get_logger

# Configure logging with desired level
configure_logging(level=logging.INFO)

# Get a logger for your module
logger = get_logger(__name__)

# Use in your code
logger.info("Starting memory profiling")
estimator = MemoryTracer(device="cuda")

# The MemoryTracer will use the configured logging system
with estimator:
    # ... (profiling code)
    pass

# All output will go through the logging system
estimator.print_memory_stats()
```

### Configuration Options

The logging system can be configured with various options:

```python
from memory_profiler import configure_logging
import logging

# Basic configuration with console output
configure_logging(level=logging.INFO)

# Log to file and console with rich formatting
configure_logging(
    level=logging.DEBUG,
    log_to_file=True,
    log_file="memory_profile.log",
    rich_format=True
)

# Disable rich formatting (for non-interactive environments)
configure_logging(level=logging.INFO, rich_format=False)
```

### Environment Variables

You can also control logging through environment variables:

```bash
# Set the default logging level
export MEMORY_PROFILER_LOG_LEVEL=10  # DEBUG=10, INFO=20, WARNING=30, etc.
```

### Instance-Specific Logging

You can set different logging levels for specific tracer instances:

```python
import logging
from memory_profiler import MemoryTracer

# Create a tracer with a specific log level
verbose_tracer = MemoryTracer(device="cuda", log_level=logging.DEBUG)
normal_tracer = MemoryTracer(device="cuda", log_level=logging.INFO)
```

## Advanced Usage & Plugins

For complex models, especially those using distributed training libraries, the built-in plugins are crucial.

*   **Distributed Training (`megatron-core`, `torch.distributed`):** The `DistributedPlugin` patches `torch.distributed` functions, and the `P2PCommunicationPlugin` patches `megatron.core.pipeline_parallel.p2p_communication` functions. This allows tracing memory in Tensor Parallel (TP) and Pipeline Parallel (PP) setups by correctly handling fake tensor propagation across ranks or pipeline stages. The plugins typically return appropriately shaped fake tensors for receive operations and skip actual communication for send operations. See `examples/mcore/pretrain_gpt.py` (`estimated_main` function) for an example using `megatron-core`.
*   **Transformer Engine:** The `TransformerEnginePlugin` patches `transformer_engine` layers and functions (like `LayerNormLinear`) to ensure they operate correctly within the fake tensor context, often by returning cloned inputs or appropriately shaped fake outputs.

Plugins are enabled by default when initializing `MemoryTracer`. If `megatron-core` or `transformer-engine` are not installed, their respective plugins will simply skip patching without error.

## How it Works

`pytorch-memory-tracing` operates through a combination of PyTorch features:

1.  **`FakeTensorMode`:** Enabled within the `MemoryTracer` context (`__enter__`/`__exit__`). This mode allows PyTorch operations to run using "fake" tensors that have metadata (shape, dtype, device) but minimal memory allocation.
2.  **`MemoryDispatchMode` (subclass of `TorchDispatchMode`):** Also enabled within the `MemoryTracer` context. This mode intercepts *all* PyTorch dispatcher calls (`aten` operations).
    *   **Memory Tracking:** For each operation, it examines the output tensors. It calculates the memory footprint based on the tensor's storage (`untyped_storage().nbytes()`) and updates the `current_memory_per_device` and `peak_memory_per_device` dictionaries.
    *   **Tensor Lifetime:** It uses `weakref.WeakKeyDictionary` (`live_tensors`) to track tensors currently allocated and `weakref.finalize` to register a callback (`release_memory`) that decrements the `current_memory_per_device` when a tensor's storage is garbage collected.
    *   **Module Attribution:** Forward pre-hooks and hooks (`register_forward_pre_hook`, `register_forward_hook`) are registered to modules. These hooks maintain a `module_stack` representing the current call hierarchy. When a tensor is created within `__torch_dispatch__`, the current module path from the stack is associated with that tensor's memory allocation (`module_memory_usage`, `tensor_to_module`).
    *   **Special Handling:** The `__torch_dispatch__` implementation includes workarounds for specific operations (e.g., `random_`, `item`, `_local_scalar_dense`) that behave poorly or error out with fake tensors, often returning dummy values or cloned inputs.
3.  **Plugin System:** Before entering the `FakeTensorMode` and `MemoryDispatchMode`, registered plugins (`TracerPlugin`) patch functions in external libraries (like `torch.distributed`, `megatron-core`, `transformer-engine`). These patches typically check if inputs are fake tensors; if so, they simulate the operation's effect on fake tensors (e.g., returning a new fake tensor for a receive operation) or become no-ops (e.g., for send operations), bypassing the original function call. The original functions are restored upon exiting the `MemoryTracer` context.

By combining these mechanisms, the library simulates the computation graph's execution flow and tracks the estimated memory allocation changes without needing to allocate the full memory or perform actual computations/communication.

## TODO
- [x] Integration to training loop of Megatron-LM
- [x] Support for TE and fused kernel in MCore
- [x] Update the logging system for better usability
- [ ] Refactor code with torch `fake_collectives.py`
- [ ] Refactor code with torch `mod_tracker.py`
- [ ] Record peak memory snapshot
- [ ] Show modulize tensor lifetime with tensor's create/save/free time.
- [ ] Build a GUI or visualization system

## Known Limitations & Considerations

*   **Estimation Accuracy:** The accuracy depends on PyTorch correctly reporting storage sizes for fake tensors and the dispatcher accurately reflecting real-world operations. Complex custom CUDA kernels might not be perfectly captured.
*   **Operator Coverage:** While common operations are handled, highly specialized or new PyTorch operations might require specific handling or patching via plugins if they interact poorly with `FakeTensorMode` or `TorchDispatchMode`.
*   **External Library Patching:** Relies on patching specific functions in external libraries. Updates to these libraries might break compatibility if function signatures or internal logic change significantly. The plugins attempt to handle missing libraries gracefully.
*   **Data-Dependent Operations:** Operations whose output shape or memory usage depends heavily on input *values* (not just shapes) can be challenging to estimate accurately with fake tensors (e.g., indexing with boolean masks).

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.
