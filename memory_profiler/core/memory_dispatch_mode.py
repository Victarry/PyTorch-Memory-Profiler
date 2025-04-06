import torch
from torch.utils._pytree import tree_map_only
import weakref
import functools
import contextlib
from collections import defaultdict

from ..utils import print_rank_0

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
        
        # List of problematic operations that need special handling
        problematic_ops = [
            "_local_scalar_dense",  # Data-dependent scalar conversion
            "random_",              # Random generation
            "item",                 # Tensor to scalar conversion
        ]

        # Check if this is an operation that we need to handle specially
        func_name = str(func)
        
        # Handle distributed operations more directly
        if any(op_name in func_name for op_name in problematic_ops):
            # For random operations
            if "random_" in func_name:
                return args[0]  # Return the input tensor unchanged
            # For scalar conversion operations
            elif "_local_scalar_dense" in func_name or "item" in func_name:
                return 42  # Return a fixed scalar value
            # Try with disabled fake tensor mode, but handle exceptions gracefully
            try:
                with torch._subclasses.fake_tensor._disable_fake_tensor_mode():
                    return func(*args, **kwargs)
            except Exception as e:
                print_rank_0(f"Handling problematic operation gracefully: {func_name}")
                if len(args) > 0 and isinstance(args[0], torch.Tensor):
                    return args[0]  # Default fallback: return the input tensor
                return None
        
        current_module = self.current_module_path
        try:
            outputs = func(*args, **kwargs)
        except Exception as e:
            print_rank_0(f"Error in {func_name}: {str(e)}")

            print(func_name, args)
            return None

        # Track all output tensors' memory usage and record their creation module
        tree_map_only(
            torch.Tensor, lambda t: self.track_tensor_memory(t, current_module), outputs
        )

        return outputs

    def track_tensor_memory(self, tensor, module_path):
        if tensor.untyped_storage() in self.live_tensors:
            return

        device_id = tensor.device.index
        if device_id is None:
            device_id = "cpu"

        # Calculate memory usage
        nbytes = tensor.untyped_storage().nbytes()

        # Update current device memory usage
        if device_id not in self.current_memory_per_device:
            self.current_memory_per_device[device_id] = 0
            self.peak_memory_per_device[device_id] = 0

        self.current_memory_per_device[device_id] += nbytes
        self.peak_memory_per_device[device_id] = max(
            self.peak_memory_per_device[device_id],
            self.current_memory_per_device[device_id],
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
            {
                "module": module_path,
                "size": nbytes,
                "shape": tensor.shape,
                "device": tensor.device,
            }
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
            handles.append(
                m.register_forward_pre_hook(pre_forward_hook, with_kwargs=True)
            )

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
            # 分组统计
            module_stats = defaultdict(
                lambda: {"count": 0, "total_size": 0, "shapes": []}
            )

            for tensor_info in tensors:
                module = tensor_info["module"]
                module_stats[module]["count"] += 1
                module_stats[module]["total_size"] += tensor_info["size"]
                module_stats[module]["shapes"].append(tensor_info["shape"])

            # 格式化报告
            report[phase] = {
                module: {
                    "count": stats["count"],
                    "total_size_mb": f"{stats['total_size'] / (1024**2):.2f} MB",
                    "shapes": stats["shapes"],
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