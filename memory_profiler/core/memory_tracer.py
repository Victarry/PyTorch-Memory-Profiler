import torch
import contextlib
from collections import defaultdict

from .memory_dispatch_mode import MemoryDispatchMode
from ..plugins import DistributedPlugin, TransformerEnginePlugin

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
        
        # 初始化插件列表
        self.plugins = []
        self.register_default_plugins()
    
    def register_default_plugins(self):
        """Register the default set of plugins."""
        self.register_plugin(DistributedPlugin())
        self.register_plugin(TransformerEnginePlugin())
    
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

    def create_fake_tensor(self, *shape, dtype=torch.float16, device=None):
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
            fake_input = torch.empty(shape, dtype=dtype, device=device)
        return fake_input

    def __enter__(self):
        """Enter the context manager, enabling both fake tensors and memory estimation."""
        # 调用所有插件的enter方法
        for plugin in self.plugins:
            plugin.enter()
        
        # 启用fake tensor mode和memory dispatch mode
        self.fake_mode.__enter__()
        self.memory_dispatch_mode.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager, disabling both fake tensors and memory estimation."""
        # 调用所有插件的exit方法
        for plugin in self.plugins:
            plugin.exit(exc_type, exc_val, exc_tb)
        
        # 退出memory dispatch mode和fake tensor mode
        self.memory_dispatch_mode.__exit__(exc_type, exc_val, exc_tb)
        return self.fake_mode.__exit__(exc_type, exc_val, exc_tb)

    def get_max_memory_allocated(self):
        """Get the maximum memory allocated during the context manager."""
        return self.memory_dispatch_mode.peak_memory_per_device

    def get_current_memory_allocated(self):
        """Get the current memory allocated during the context manager."""
        # Convert device_id from numeric keys (0, 1, etc.) to string keys ('cuda:0', 'cuda:1', etc.)
        # to ensure consistent access and avoid KeyError
        current_memory = {}
        for device_id, memory in self.memory_dispatch_mode.current_memory_per_device.items():
            if isinstance(device_id, int) or (isinstance(device_id, str) and device_id.isdigit()):
                device_key = f"cuda:{device_id}"
            else:
                device_key = device_id
            current_memory[device_key] = memory
            
        # Also add a fallback numeric index access for backward compatibility
        for i, (device_id, memory) in enumerate(self.memory_dispatch_mode.current_memory_per_device.items()):
            current_memory[i] = memory
            
        return current_memory

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