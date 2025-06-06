import torch
import contextlib
from collections import defaultdict
import logging

from .memory_dispatch_mode import MemoryDispatchMode, ProblematicOpsDispatchMode
from .mod_tracker import ModTracker
from ..plugins import (
    DistributedPlugin,
    TransformerEnginePlugin,
    P2PCommunicationPlugin,
    MegatronCorePlugin,
)
from .logger import get_logger, log_memory_table

# Get a logger for this module
logger = get_logger(__name__)


class MemoryTracer:
    """
    Memory tracing utility that uses fake tensors to estimate memory usage.
    """

    def __init__(self, device="cuda", log_level=None):
        """
        Initialize the MemoryTracer.

        Args:
            device (str): The device to emulate, default is "cuda"
            log_level (int, optional): Logging level to use for this tracer
        """
        self.device = device
        self.mod_tracker = ModTracker()
        self.problematic_ops_mode = ProblematicOpsDispatchMode(log_level=log_level)
        self.memory_dispatch_mode = MemoryDispatchMode(log_level=log_level, mod_tracker=self.mod_tracker)
        
        # Configure specific log level for this tracer if provided
        if log_level is not None:
            self.logger = get_logger(f"{__name__}.instance_{id(self)}")
            self.logger.setLevel(log_level)
        else:
            self.logger = logger

        try:
            from torch._subclasses import FakeTensorMode

            self.fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        except ImportError:
            raise ImportError(
                "FakeTensorMode not found. Please use PyTorch version >= 1.12.0"
            )

        self.op_counts = defaultdict(int)
        self.op_memory = defaultdict(int)
        self.phase_memory = defaultdict(int)
        self.phase_memory_snapshots = defaultdict(lambda: {"before": {}, "after": {}})
        self.current_phase = "initialization"

        self.plugins = []
        self.register_default_plugins()

    def register_default_plugins(self):
        """Register the default set of plugins."""
        self.register_plugin(DistributedPlugin())
        self.register_plugin(TransformerEnginePlugin())
        self.register_plugin(P2PCommunicationPlugin())
        self.register_plugin(MegatronCorePlugin())

    def register_plugin(self, plugin):
        """Register a plugin with the tracer."""
        plugin.setup(self)
        self.plugins.append(plugin)

    def create_fake_tensor_from_tensor(self, tensor, device=None):
        """
        Convert a real tensor to a fake tensor.

        Args:
            tensor (torch.Tensor): The tensor to convert
            device (str, optional): The device to place the fake tensor on

        Returns:
            torch.Tensor: A fake tensor with the same properties
        """
        if device is None:
            device = self.device

        with self.fake_mode:
            fake_tensor = torch.empty(
                tensor.shape,
                dtype=tensor.dtype,
                device=device,
                requires_grad=tensor.requires_grad,
            )
        return fake_tensor

    def create_fake_tensor(
        self, *shape, dtype=torch.float16, device=None, requires_grad=False
    ):
        """
        Create a fake batch of input data.

        Args:
            shape: Shape of the fake tensor
            dtype (torch.dtype): Data type
            device (str, optional): Device to place the fake tensor on

        Returns:
            torch.Tensor: A fake input tensor
        """
        if device is None:
            device = self.device

        with self.fake_mode:
            fake_input = torch.empty(
                shape, dtype=dtype, device=device, requires_grad=requires_grad
            )
        return fake_input

    def __enter__(self):
        """Enter the context manager, enabling both fake tensors and memory estimation."""
        self.mod_tracker.__enter__()
        for plugin in self.plugins:
            plugin.enter()

        self.fake_mode.__enter__()
        self.problematic_ops_mode.__enter__()
        self.memory_dispatch_mode.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager, disabling both fake tensors and memory estimation."""
        self.memory_dispatch_mode.__exit__(exc_type, exc_val, exc_tb)
        self.problematic_ops_mode.__exit__(exc_type, exc_val, exc_tb)
        self.fake_mode.__exit__(exc_type, exc_val, exc_tb)
        
        for plugin in self.plugins:
            plugin.exit(exc_type, exc_val, exc_tb)
        self.mod_tracker.__exit__(exc_type, exc_val, exc_tb)
        return False

    def get_max_memory_allocated(self):
        """Get the maximum memory allocated during the context manager."""
        max_memory_mb = {}
        for (
            device_id,
            memory_bytes,
        ) in self.memory_dispatch_mode.peak_memory_per_device.items():
            max_memory_mb[device_id] = memory_bytes / (
                1024 * 1024
            )  # Convert bytes to MB
        return max_memory_mb

    def get_current_memory_allocated(self):
        """Get the current memory allocated during the context manager."""
        # Convert device_id from numeric keys (0, 1, etc.) to string keys ('cuda:0', 'cuda:1', etc.)
        # to ensure consistent access and avoid KeyError
        current_memory = {}
        for (
            device_id,
            memory,
        ) in self.memory_dispatch_mode.current_memory_per_device.items():
            current_memory[device_id] = memory

        current_memory_mb = {}
        for device_id, memory_bytes in current_memory.items():
            current_memory_mb[device_id] = memory_bytes / (
                1024 * 1024
            )  # Convert bytes to MB
        return current_memory_mb

    @contextlib.contextmanager
    def track_phase(self, phase_name):
        """
        Context manager to track memory usage during different phases of training.

        Args:
            phase_name (str): Name of the phase (e.g., "forward", "backward", "optimizer")
        """
        # Record memory before entering the phase
        self.phase_memory_snapshots[phase_name]["before"] = self.get_max_memory_allocated()

        prev_phase = self.current_phase
        self.current_phase = phase_name

        self.memory_dispatch_mode.current_phase = phase_name

        try:
            yield
        finally:
            # Record memory after exiting the phase
            self.phase_memory_snapshots[phase_name]["after"] = self.get_max_memory_allocated()
            self.current_phase = prev_phase
            self.memory_dispatch_mode.current_phase = prev_phase

    def get_memory_stats(self):
        """
        Get comprehensive memory statistics.

        Returns:
            dict: Dictionary containing peak memory, per-module memory, and per-phase memory
        """
        stats = {
            "peak_memory": {
                f"device_{dev}": f"{mem / (1024**2):.2f} MB"
                for dev, mem in self.memory_dispatch_mode.peak_memory_per_device.items()
            },
            "current_memory": {
                f"device_{dev}": f"{mem / (1024**2):.2f} MB"
                for dev, mem in self.memory_dispatch_mode.current_memory_per_device.items()
            },
            "module_memory": self.memory_dispatch_mode.get_module_memory_report(),
            "phase_memory": {
                phase: f"{mem / (1024**2):.2f} MB"
                for phase, mem in self.phase_memory.items()
            },
            "phase_memory_snapshots": {
                phase: {
                    "before": {dev: f"{mem:.2f} MB" for dev, mem in data["before"].items()},
                    "after": {dev: f"{mem:.2f} MB" for dev, mem in data["after"].items()},
                }
                for phase, data in self.phase_memory_snapshots.items()
            },
            "operation_counts": dict(self.op_counts),
        }
        return stats

    def print_memory_stats(self, header=None):
        """
        Print memory statistics in a readable format using tables.

        Args:
            header (str, optional): Optional header text for the memory stats
        """
        stats = self.get_memory_stats()

        # Log the memory usage header
        header_text = f"MEMORY USAGE ESTIMATION: {header if header else ''}"
        self.logger.info(f"\n===== {header_text} =====")

        # --- Peak Memory Usage Table ---
        rows = []
        for dev, mem in stats["peak_memory"].items():
            mem_val = float(mem.split()[0])
            rows.append([dev, f"{mem_val:.2f}"])
        
        log_memory_table(
            self.logger,
            "Peak Memory Usage:",
            ["Device", "Peak Memory (MB)"],
            rows,
            level=logging.CRITICAL
        )

        # --- Current Memory Usage Table ---
        rows = []
        for dev, mem in stats["current_memory"].items():
            mem_val = float(mem.split()[0])
            rows.append([dev, f"{mem_val:.2f}"])
            
        log_memory_table(
            self.logger,
            "Current Memory Usage:",
            ["Device", "Current Memory (MB)"],
            rows,
            level=logging.CRITICAL
        )

        # --- Phase Memory Changes Table ---
        rows = []
        for phase, snapshots in stats["phase_memory_snapshots"].items():
            before_mem = snapshots["before"]
            after_mem = snapshots["after"]
            
            # Get all unique devices across before and after
            all_devices = set(list(before_mem.keys()) + list(after_mem.keys()))
            
            for device in all_devices:
                before_val = float(before_mem.get(device, "0 MB").split()[0])
                after_val = float(after_mem.get(device, "0 MB").split()[0])
                delta = after_val - before_val
                
                # Only show CPU memory if it's >= 100 MB, always show CUDA memory
                device_str = str(device)
                is_cuda = 'cuda' in device_str.lower()
                if is_cuda or (not is_cuda and (before_val >= 100 or after_val >= 100)):
                    rows.append([phase, device_str, f"{before_val:.2f}", f"{after_val:.2f}", f"{delta:.2f}"])
        
        log_memory_table(
            self.logger,
            "Peak Memory Changes in each phase:",
            ["Phase", "Device", "Before (MB)", "After (MB)", "Delta (MB)"],
            rows,
            level=logging.CRITICAL
        )
