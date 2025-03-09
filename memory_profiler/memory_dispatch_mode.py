import torch
from torch.utils._pytree import tree_map_only
import weakref
import functools
import contextlib
from collections import defaultdict


class MemoryDispatchMode(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self):
        self.peak_memory_per_device = {}
        self.current_memory_per_device = {}
        self.live_tensors = weakref.WeakKeyDictionary()

        self.module_stack = []
        self.module_memory_usage = defaultdict(lambda: defaultdict(int))
        self.tensor_to_module = weakref.WeakKeyDictionary()

        self.phase_tensor_modules = defaultdict(list)
        self.current_phase = "initialization"
    
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs is not None else {}

        current_module = self.current_module_path
        outputs = func(*args, **kwargs)

        # Track all output tensors' memory usage and record their creation module
        tree_map_only(torch.Tensor, lambda t: self.track_tensor_memory(t, current_module), outputs)

        return outputs

    def track_tensor_memory(self, tensor, module_path):
        if tensor.untyped_storage() in self.live_tensors:
            return

        device_id = tensor.device.index
        if device_id is None:
            device_id = 'cpu'

        # Calculate memory usage
        nbytes = tensor.untyped_storage().nbytes()

        # Update current device memory usage
        if device_id not in self.current_memory_per_device:
            self.current_memory_per_device[device_id] = 0
            self.peak_memory_per_device[device_id] = 0

        self.current_memory_per_device[device_id] += nbytes
        self.peak_memory_per_device[device_id] = max(
            self.peak_memory_per_device[device_id], self.current_memory_per_device[device_id]
        )

        # Record which module created this tensor
        self.tensor_to_module[tensor.untyped_storage()] = module_path

        # Update module memory usage statistics
        self.module_memory_usage[module_path][device_id] += nbytes

        # Mark as tracked
        self.live_tensors[tensor.untyped_storage()] = True

        # When the tensor is released, reduce the memory count
        weakref.finalize(
            tensor.untyped_storage(),
            functools.partial(self.release_memory, device_id, nbytes, module_path),
        )
        # Record which phase the tensor was created in, and which module it belongs to
        self.phase_tensor_modules[self.current_phase].append(
            {'module': module_path, 'size': nbytes, 'shape': tensor.shape, 'device': tensor.device}
        )

    def make_module_pre_forward_hook(self):
        """Create a forward hook for tracking module execution
        The hook is called before the forward pass of the module"""

        def hook(module, inputs, kargs):
            module_path = f"{module.__class__.__name__}"
            if self.module_stack:
                parent_path = self.module_stack[-1]
                module_path = f"{parent_path}.{module_path}"

            # Push the current module to stack before processing outputs
            self.module_stack.append(module_path)

        return hook

    def make_module_forward_hook(self):
        """Create a forward hook for tracking module execution
        The hook will be called every time after forward() has computed an output."""

        def hook(module, inputs, outputs):
            self.module_stack.pop()

        return hook

    def register_hooks_to_module(self, module):
        """Register forward hooks to the given module and all its submodules"""
        pre_forward_hook = self.make_module_pre_forward_hook()
        forward_hook = self.make_module_forward_hook()
        handles = []

        # Register hook to the main module and all submodules
        for m in [module] + list(module.modules()):
            handles.append(m.register_forward_hook(forward_hook))
            handles.append(m.register_forward_pre_hook(pre_forward_hook, with_kwargs=True))

        return handles

    def remove_hooks(self, handles):
        """Remove all registered hooks using their handles"""
        for handle in handles:
            handle.remove()

    @property
    def current_module_path(self):
        """Get the current module path"""
        if not self.module_stack:
            return "Unknown"
        return self.module_stack[-1]

    def release_memory(self, device_id, nbytes, module_path):
        """Called when a tensor is released"""
        self.current_memory_per_device[device_id] -= nbytes

    def get_module_memory_report(self):
        """Generate a report of memory usage for each module"""
        report = {}
        for module_path, device_usage in self.module_memory_usage.items():
            report[module_path] = {
                f"device_{dev}": f"{usage / (1024**2):.2f} MB"
                for dev, usage in device_usage.items()
            }
        return report

    def get_phase_tensor_report(self):
        """Generate a detailed report of tensors created in each phase"""
        report = {}
        for phase, tensors in self.phase_tensor_modules.items():
            # 按module分组统计
            module_stats = defaultdict(lambda: {'count': 0, 'total_size': 0, 'shapes': []})

            for tensor_info in tensors:
                module = tensor_info['module']
                module_stats[module]['count'] += 1
                module_stats[module]['total_size'] += tensor_info['size']
                module_stats[module]['shapes'].append(tensor_info['shape'])

            # 格式化报告
            report[phase] = {
                module: {
                    'count': stats['count'],
                    'total_size_mb': f"{stats['total_size'] / (1024**2):.2f} MB",
                    'shapes': stats['shapes'],
                }
                for module, stats in module_stats.items()
            }

        return report

    @contextlib.contextmanager
    def trace_module(self, name):
        """Context manager for manually tracking non-module code blocks"""
        self.module_stack.append(name)
        try:
            yield
        finally:
            self.module_stack.pop()


class MemoryEstimator:

    def __init__(self, device="cuda"):
        """
        Initialize the MemoryEstimator.

        Args:
            device (str): The device to emulate, default is "cuda"
        """
        self.device = device
        self.memory_dispatch_mode = MemoryDispatchMode()
        try:
            from torch._subclasses import FakeTensorMode

            self.fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        except ImportError:
            raise ImportError("FakeTensorMode not found. Please use PyTorch version >= 1.12.0")

        self.op_counts = defaultdict(int)
        self.op_memory = defaultdict(int)
        self.phase_memory = defaultdict(int)
        self.current_phase = "initialization"
    
    def to_fake_tensor(self, tensor, device=None):
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
                tensor.shape, dtype=tensor.dtype, device=device, requires_grad=tensor.requires_grad
            )
        return fake_tensor

    def create_fake_tensor(
        self, *shape, dtype=torch.float16, device=None
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
                shape, dtype=dtype, device=device
            )
        return fake_input

    def __enter__(self):
        """Enter the context manager, enabling both fake tensors and memory estimation."""
        self.fake_mode.__enter__()
        self.memory_dispatch_mode.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager, disabling both fake tensors and memory estimation."""
        self.memory_dispatch_mode.__exit__(exc_type, exc_val, exc_tb)
        return self.fake_mode.__exit__(exc_type, exc_val, exc_tb)
    
    def get_max_memory_allocated(self):
        """Get the maximum memory allocated during the context manager."""
        return self.memory_dispatch_mode.peak_memory_per_device

    def get_current_memory_allocated(self):
        """Get the current memory allocated during the context manager."""
        return self.memory_dispatch_mode.current_memory_per_device

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
                phase: f"{mem / (1024**2):.2f} MB" for phase, mem in self.phase_memory.items()
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
