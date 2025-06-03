# PyTorch Memory Tracing

A utility library designed to estimate and analyze the memory footprint of PyTorch models, particularly large models, often without requiring actual GPU resources. This tool leverages PyTorch's `FakeTensorMode` and `TorchDispatchMode` capabilities to simulate model execution and track estimated memory allocation.

## Features
*   **Memory Tracing with FakeTensor:** Estimate peak and current memory usage for model training steps (forward, backward, optimizer) using fake tensors without actual execution.
*   **Distributed Training Support:** Support distributed training memory tracer with one single process.
*   **Module-Level Tracking:** Pinpoint memory usage to specific modules within your model hierarchy using PyTorch hooks.
*   **Detailed Visualization Reporting:** Generate reports on peak/current memory, memory usage per module, memory usage per phase, and tensor creation statistics.
*   **Integration with Megatron-LM:** Seamless memory tracing your training scripts with minimal changes.

## Installation
```bash
# Clone the repository
git clone https://github.com/Victarry/PyTorch-Memory-Profiler.git
cd PyTorch-Memory-Profiler && pip install .
```

## Quick Start for Megatron-LM

Here's how to integrate memory tracing with Megatron-LM training:

### Megatron-LM setup
* **Method 1:** 
Clone the specific Megatron-LM branch for memory-tracing:
```bash
git clone https://github.com/Victarry/Megatron-LM.git
cd megatron-lm && git checkout denliu/patch_for_memory_tracing
```

* **Method 2:**
Apply the patch to your existing Megatron-LM installation:
```bash
cd YOUR_MEGATRON_PATH
git apply /path/to/PyTorch-Memory-Profiler/patches/memory_tracing_mcore.patch
```

### Memory tracing with existing training scripts

1. Enable memory tracing by adding the `--memory-tracing` flag to your existing Megatron-LM training command.
2. Setting the the `WORLD_RANK` and `RANK` to specifiy total number of GPUs and current rank ID.
3. [Optional] Set `--save-peak-memory-snapshot` to specify the location of saved peak memory snapshot. The snapthot is saved as json format, you can directly check the tensor information or using provides visualizer.

Explore ./examples/megatron-lm for predefined scripts.

**Key points:**
- Training will automatically stop after 2 iterations when memory tracing is enabled
- Mock data is used automatically, so no real dataset is required
- The tool works with distributed training configurations (TP, PP, EP)
- Only one process is needed to lauch for memory profiling, no `torchrun` is needed even for distributed training.
- Token-drop-and-pad training is forcely applied for MoE training scripts.

### Visualization of peak memory snapshot

After running memory tracing, visualize the results using the included Streamlit-based visualizer:

```bash
# Navigate to tools directory
cd PyTorch-Memory-Profiler/tools

# Install visualization dependencies
uv sync

# Launch the visualizer
uv run streamlit run visualizer.py -- --file /path/to/memory_snapshot.json
```

The visualizer provides several interactive views:

1. **Phase Breakdown**: Memory usage by execution phase (setup, forward, backward, optimizer)
2. **Module Hierarchy**: Hierarchical view of memory usage by PyTorch modules
3. **Merged Modules**: Similar modules (like transformer layers) grouped together
4. **Stack Trace Analysis**: Tensors grouped by their creation location in code
5. **Tensor Details**: Detailed information about individual tensors

**Features:**
- Interactive filtering and sorting
- Memory usage charts with hover details
- Module pattern detection (e.g., `layers.0`, `layers.1` → `layers.*`)
- Export capabilities for further analysis
- Multi-device support for distributed training analysis

TODO: Detailed explanation on the visualization items.

## Documentation

For more detailed usage instructions, see:
- [Custom Usage](docs/custom-usage.md) - Module tracking, phase tracking, and memory snapshot analysis

## TODO
- [ ] FP8 support in TransformerEngine

## Known Limitations & Considerations

*   **Estimation Accuracy:** The accuracy depends on PyTorch correctly reporting storage sizes for fake tensors and the dispatcher accurately reflecting real-world operations. Complex custom CUDA kernels might not be perfectly captured.
*   **Operator Coverage:** While common operations are handled, highly specialized or new PyTorch operations might require specific handling or patching via plugins if they interact poorly with `FakeTensorMode` or `TorchDispatchMode`.
*   **External Library Patching:** Relies on patching specific functions in external libraries. Updates to these libraries might break compatibility if function signatures or internal logic change significantly. The plugins attempt to handle missing libraries gracefully.
*   **Data-Dependent Operations:** Operations whose output shape or memory usage depends heavily on input *values* (not just shapes) is not supported with fake tensors (e.g., indexing with boolean masks).
*   **FP8 Support:** Currently, FP8 operations are not supported in memory tracing mode due to limitations in fake tensor handling.
