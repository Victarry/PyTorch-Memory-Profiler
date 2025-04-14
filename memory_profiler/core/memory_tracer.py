import torch
import contextlib
from collections import defaultdict

from .memory_dispatch_mode import MemoryDispatchMode
from ..plugins import DistributedPlugin, TransformerEnginePlugin, P2PCommunicationPlugin


class MemoryTracer:
    """
    Memory tracing utility that uses fake tensors to estimate memory usage.
    """

    def __init__(self, device="cuda"):
        """
        Initialize the MemoryTracer.

        Args:
            device (str): The device to emulate, default is "cuda"
        """
        self.device = device
        self.memory_dispatch_mode = MemoryDispatchMode()
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
        self.current_phase = "initialization"

        self.plugins = []
        self.register_default_plugins()

    def register_default_plugins(self):
        """Register the default set of plugins."""
        self.register_plugin(DistributedPlugin())
        self.register_plugin(TransformerEnginePlugin())
        self.register_plugin(P2PCommunicationPlugin())

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
        for plugin in self.plugins:
            plugin.enter()

        self.fake_mode.__enter__()
        self.memory_dispatch_mode.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager, disabling both fake tensors and memory estimation."""
        for plugin in self.plugins:
            plugin.exit(exc_type, exc_val, exc_tb)

        self.memory_dispatch_mode.__exit__(exc_type, exc_val, exc_tb)
        return self.fake_mode.__exit__(exc_type, exc_val, exc_tb)

    def get_max_memory_allocated(self):
        """Get the maximum memory allocated during the context manager."""
        max_memory_mb = {}
        for device_id, memory_bytes in self.memory_dispatch_mode.peak_memory_per_device.items():
            max_memory_mb[device_id] = memory_bytes / (1024 * 1024)  # Convert bytes to MB
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
            current_memory_mb[device_id] = memory_bytes / (1024 * 1024)  # Convert bytes to MB
        return current_memory_mb

    @contextlib.contextmanager
    def track_phase(self, phase_name):
        """
        Context manager to track memory usage during different phases of training.

        Args:
            phase_name (str): Name of the phase (e.g., "forward", "backward", "optimizer")
        """
        prev_phase = self.current_phase
        self.current_phase = phase_name

        self.memory_dispatch_mode.current_phase = phase_name

        try:
            yield
        finally:
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
            "operation_counts": dict(self.op_counts),
        }
        return stats

    def print_memory_stats(self, detailed=False):
        """
        Print memory statistics in a readable format.

        Args:
            detailed (bool): Whether to print detailed per-module statistics
        """
        stats = self.get_memory_stats()

        print("\n===== MEMORY USAGE ESTIMATION =====")
        print("\nPeak Memory Usage:")
        for dev, mem in stats["peak_memory"].items():
            print(f"  {dev}: {mem}")

        print("\nCurrent Memory Usage:")
        for dev, mem in stats["current_memory"].items():
            print(f"  {dev}: {mem}")

        print("\nMemory Usage by Phase:")
        for phase, mem in stats["phase_memory"].items():
            print(f"  {phase}: {mem}")

        if detailed:
            print("\nMemory Usage by Module:")
            for module, device_usage in stats["module_memory"].items():
                print(f"  {module}:")
                for dev, mem in device_usage.items():
                    print(f"    {dev}: {mem}")

            print("\nOperation Counts:")
            for op, count in stats["operation_counts"].items():
                print(f"  {op}: {count}")

        print("\n===== TENSOR CREATION BY PHASE =====")
        phase_tensor_report = self.memory_dispatch_mode.get_phase_tensor_report()
        for phase, modules in phase_tensor_report.items():
            print(f"\nPhase: {phase}")
            for module, info in modules.items():
                print(f"  Module: {module}")
                print(f"    Tensor count: {info['count']}")
                print(f"    Total size: {info['total_size_mb']}")
                if detailed:
                    print(f"    Tensor shapes: {info['shapes']}")
