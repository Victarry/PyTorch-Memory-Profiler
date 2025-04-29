import torch
from torch.utils._pytree import tree_map_only
import weakref
import functools
import contextlib
from collections import defaultdict
from .logger import get_logger, log_memory_table

# Get a logger for this module
logger = get_logger(__name__)

class MemoryDispatchMode(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self, log_level=None):
        self.peak_memory_per_device = {}
        self.current_memory_per_device = {}
        # Stores {storage: {'size': nbytes, 'shape': shape, 'module': module_path}}
        self.live_tensors = weakref.WeakKeyDictionary()

        self.module_stack = []
        self.module_memory_usage = defaultdict(lambda: defaultdict(int))

        self.phase_tensor_modules = defaultdict(list)
        self.current_phase = "initialization"
        
        # Configure specific log level for this instance if provided
        if log_level is not None:
            self.logger = get_logger(f"{__name__}.instance_{id(self)}")
            self.logger.setLevel(log_level)
        else:
            self.logger = logger

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
        
        # --- Patch for aten.split_with_sizes.default with FakeTensor ---
        if func_name == "aten.split_with_sizes.default":
            if len(args) == 3:
                input_tensor, split_sizes, dim = args
            elif len(args) == 2:
                input_tensor, split_sizes = args
                dim = kwargs.get('dim', 0)
            else:
                logger.error(f"Unexpected number of arguments for aten.split_with_sizes.default: {args}")

            if isinstance(input_tensor, torch._subclasses.fake_tensor.FakeTensor):
                num_chunks = len(split_sizes)
                self.logger.debug(f"Patching aten.split_with_sizes for FakeTensor. Using torch.chunk with {num_chunks} chunks along dim {dim}.")
                # Use torch.chunk logic for uniform splitting
                outputs = torch.chunk(input_tensor, num_chunks, dim=dim)
                # Ensure outputs are tracked and potentially moved to CUDA like other outputs
                tree_map_only(
                    torch.Tensor, lambda t: self.track_tensor_memory(t, self.current_module_path), outputs
                )
                outputs_mod = []
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                         outputs_mod.append(output.to('cuda'))
                    else:
                        outputs_mod.append(output)
                return tuple(outputs_mod)
        # --- End Patch ---

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
                self.logger.debug(f"Handling problematic operation gracefully: {func_name}")
                if len(args) > 0 and isinstance(args[0], torch.Tensor):
                    return args[0]  # Default fallback: return the input tensor
                return None
        
        current_module = self.current_module_path
        try:
            outputs = func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in {func_name}: {str(e)}")
            self.logger.error(f"{func_name}, {args}")
            import traceback
            import sys
            
            # Get stack trace using format_stack
            stack_trace = traceback.format_stack()
            
            # Format and log the stack trace
            self.logger.error(f"Stack trace (top level first):\n{''.join(stack_trace)}")
            
            # Also log the exception info if available
            if sys.exc_info()[1] is not None:
                self.logger.error(f"Exception: {type(e).__name__}: {str(e)}")
            
            return None

        # Track all output tensors' memory usage and record their creation module
        tree_map_only(
            torch.Tensor, lambda t: self.track_tensor_memory(t, current_module), outputs
        )
        if isinstance(outputs, torch.Tensor):
            # Always move to cuda since modules.to("cuda") doesn't work for FakeTensor.
            return outputs.to('cuda') 
        elif isinstance(outputs, tuple):
            outputs_mod = []
            for output in outputs:
                if isinstance(output, torch.Tensor):
                     outputs_mod.append(output.to('cuda'))
                else:
                    outputs_mod.append(output)
            return tuple(outputs_mod)
        else:
            return outputs

    # TODO: revise this implementation
    def create_patched_resize_(self, nbytes, device_id):
        def patched_resize_(new_size_bytes, *resize_args, **resize_kwargs):
            # Check if resizing to zero and if we should handle the release
            try:
                # Call the original resize first
                self.release_memory(device_id, nbytes - new_size_bytes)
                # storage.resize_(new_size_bytes, *resize_args, **resize_kwargs)
            except Exception as e:
                self.logger.error(f"Error during patched resize_ for storage: {e}")
                # If original resize failed, re-raise the exception
                raise e
        return patched_resize_

    def track_tensor_memory(self, tensor, module_path):
        storage = tensor.untyped_storage()
        if storage in self.live_tensors:
            return
        device_id = tensor.device

        # Calculate memory usage
        nbytes = storage.nbytes()

        # Update current device memory usage
        if device_id not in self.current_memory_per_device:
            self.current_memory_per_device[device_id] = 0
            self.peak_memory_per_device[device_id] = 0

        self.current_memory_per_device[device_id] += nbytes
        self.peak_memory_per_device[device_id] = max(
            self.peak_memory_per_device[device_id],
            self.current_memory_per_device[device_id],
        )

        # Update module memory usage statistics
        self.module_memory_usage[module_path][device_id] += nbytes

        # Store tensor info keyed by storage
        self.live_tensors[storage] = {
            "size": nbytes,
            "shape": tensor.shape,
            "module": module_path,
            "device": device_id, # Store device for release
            "phase": self.current_phase # Store creation phase
        }

        # # --- Patch storage.resize_ to intercept manual release --- 
        storage.resize_ = self.create_patched_resize_(nbytes, device_id)

        # Apply the patch - monkey patching the storage instance
        # This assumes storage objects are mutable and allow attribute assignment
        # storage.resize_ = patched_resize_
        # --- End Patch ---
        weakref.finalize(
            storage,
            functools.partial(self.release_memory, device_id, nbytes),
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

    def release_memory(self, device_id, nbytes):
        """Helper method to decrease memory count - called by finalize or patched resize_(0)."""
        if device_id in self.current_memory_per_device:
             self.current_memory_per_device[device_id] -= nbytes
        # Entry in self.live_tensors is removed automatically by WeakKeyDictionary

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
            module_stats = defaultdict(
                lambda: {"count": 0, "total_size": 0, "shapes": []}
            )

            for tensor_info in tensors:
                module = tensor_info["module"]
                module_stats[module]["count"] += 1
                module_stats[module]["total_size"] += tensor_info["size"]
                module_stats[module]["shapes"].append(tensor_info["shape"])

            report[phase] = {
                module: {
                    "count": stats["count"],
                    "total_size_mb": f"{stats['total_size'] / (1024**2):.2f} MB",
                    "shapes": stats["shapes"],
                }
                for module, stats in module_stats.items()
            }

        return report

    def log_live_tensors(self, min_memory_mb=0):
        """Logs a table of currently live tensors, sorted by size.

        Args:
            min_memory_mb (float): Minimum memory size (in MB) for a tensor to be included in the report.
                                   Defaults to 0 (show all).
        """
        live_tensor_info = []
        total_displayed_memory = 0
        min_memory_bytes = min_memory_mb * (1024**2)

        # Iterate through the WeakKeyDictionary safely
        # .items() might be unsafe if collection happens during iteration
        # Create a temporary list of items first
        current_live_items = list(self.live_tensors.items())

        for storage, info in current_live_items:
            # Check if storage is still valid (though WeakKeyDict handles this mostly)
            # This check might be redundant but adds safety
            try:
                # Attempt a cheap operation to check validity implicitly
                _ = storage.nbytes() 
                live_tensor_info.append({
                    "size": info["size"],
                    "shape": info["shape"],
                    "module": info["module"],
                    "phase": info["phase"] # Extract phase
                })
                total_displayed_memory += info["size"]
            except RuntimeError: 
                # Storage might have been invalidated between list creation and access
                continue 

        # Filter by minimum memory size
        filtered_live_tensor_info = [
            info for info in live_tensor_info if info["size"] >= min_memory_bytes
        ]

        # Sort by size descending
        filtered_live_tensor_info.sort(key=lambda x: x["size"], reverse=True)

        # Format and log the table
        rows = []
        for info in filtered_live_tensor_info:
            module_str = ".".join(info['module'].split('.')[-5:])
            phase_str = info['phase'][:18] + '..' if len(info['phase']) > 20 else info['phase']
            size_mb = info['size'] / (1024**2)
            shape_str = str(info['shape'])
            rows.append([phase_str, module_str, f"{size_mb:.2f}", shape_str])
            
        # Log the memory table
        log_memory_table(
            self.logger,
            f"Live Tensors Report (Minimum Size: {min_memory_mb} MB):",
            ["Phase", "Module", "Size (MB)", "Shape"],
            rows
        )
        
        # Log the total
        total_displayed_mb = total_displayed_memory / (1024**2)
        self.logger.info(f"Total Displayed Live Tensor Memory: {total_displayed_mb:.2f} MB")

    @contextlib.contextmanager
    def trace_module(self, name):
        """Context manager for manually tracking non-module code blocks"""
        self.module_stack.append(name)
        try:
            yield
        finally:
            self.module_stack.pop() 