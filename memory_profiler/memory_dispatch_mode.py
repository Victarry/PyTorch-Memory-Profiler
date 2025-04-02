import torch
from torch.utils._pytree import tree_map_only
import weakref
import functools
import contextlib
from collections import defaultdict

def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message)
    else:
        print(message)

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
            # 按module分组统计
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


class MemoryTracer:

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
        # Patch torch.distributed functions to handle fake tensors
        if hasattr(torch, 'distributed') and torch.distributed.is_available():
            # Import FakeTensor for proper type checking
            from torch._subclasses.fake_tensor import FakeTensor
            
            original_funcs = {}
            
            # Get all distributed functions that might need patching
            dist_functions = [name for name in dir(torch.distributed) 
                             if callable(getattr(torch.distributed, name)) 
                             and not name.startswith('_')]
            
            # Create patches for each distributed function
            for func_name in dist_functions:
                original_func = getattr(torch.distributed, func_name)
                original_funcs[func_name] = original_func
                
                # Define the patched function
                def make_patched_dist_func(orig_func):
                    @functools.wraps(orig_func)
                    def patched_dist_func(*args, **kwargs):
                        # Check if any of the arguments are fake tensors
                        has_fake_tensor = any(
                            isinstance(arg, FakeTensor)
                            for arg in args
                            if isinstance(arg, torch.Tensor)
                        )
                        
                        # Also check in kwargs
                        has_fake_tensor = has_fake_tensor or any(
                            isinstance(arg, FakeTensor)
                            for arg in kwargs.values()
                            if isinstance(arg, torch.Tensor)
                        )
                        
                        if has_fake_tensor:
                            # TODO: returned tensor shape should be based on function type
                            # If there are fake tensors, return copies of input tensors
                            if len(args) > 0 and isinstance(args[0], torch.Tensor):
                                output = args[0].clone()
                                output.wait = lambda: None
                                return output
                            # Check if there's a tensor in kwargs to return
                            for arg in kwargs.values():
                                if isinstance(arg, torch.Tensor):
                                    return arg.clone()
                            # If no tensor found, just return None
                            return None
                        
                        # Otherwise, call the original function
                        return orig_func(*args, **kwargs)
                    
                    return patched_dist_func
                
                # Set the patched function
                setattr(torch.distributed, func_name, make_patched_dist_func(original_func))
            
            # Store original functions to restore them on exit
            self._original_dist_funcs = original_funcs
        
        # Patch transformer_engine operations to handle fake tensors
        try:
            import transformer_engine
            import transformer_engine.pytorch.ops as te_ops
            from torch._subclasses.fake_tensor import FakeTensor
            
            # Store original functions for later restoration
            self._original_te_funcs = {}
            
            # Patch forward methods on transformer_engine module ops
            for op_name in dir(te_ops):
                op_obj = getattr(te_ops, op_name)
                if hasattr(op_obj, 'forward') and callable(getattr(op_obj, 'forward')):
                    try:
                        orig_forward = getattr(op_obj, 'forward')
                        self._original_te_funcs[f"{op_name}.forward"] = orig_forward
                        
                        # Create patched forward method
                        def make_patched_te_forward(orig_func, op_name):
                            @functools.wraps(orig_func)
                            def patched_forward(self, *args, **kwargs):
                                # Check if any argument is a fake tensor
                                has_fake = any(
                                    isinstance(arg, FakeTensor)
                                    for arg in args
                                    if isinstance(arg, torch.Tensor)
                                )
                                
                                has_fake = has_fake or any(
                                    isinstance(arg, FakeTensor)
                                    for arg in kwargs.values()
                                    if isinstance(arg, torch.Tensor)
                                )
                                
                                if has_fake:
                                    print_rank_0(f"Handling fake tensor in {op_name}")
                                    # For LayerNorm, return input with same shape 
                                    if "LayerNorm" in op_name:
                                        if len(args) > 0 and isinstance(args[0], torch.Tensor):
                                            return args[0].clone()
                                    
                                    # For other operations, try to determine output shape
                                    # based on operation type and input shapes
                                    for arg in args:
                                        if isinstance(arg, torch.Tensor):
                                            return arg.clone()
                                    
                                    # Fallback
                                    for arg in kwargs.values():
                                        if isinstance(arg, torch.Tensor):
                                            return arg.clone()
                                    
                                    return None
                                
                                return orig_func(self, *args, **kwargs)
                            
                            return patched_forward
                        
                        # Apply the patch
                        setattr(op_obj, 'forward', make_patched_te_forward(orig_forward, op_name))
                    except Exception as e:
                        print_rank_0(f"Failed to patch {op_name}: {e}")
                
            # Patch specific problematic functions in transformer_engine
            if hasattr(transformer_engine.pytorch.ops.basic.layer_norm, 'layernorm_fwd'):
                orig_layernorm_fwd = transformer_engine.pytorch.ops.basic.layer_norm.layernorm_fwd
                self._original_te_funcs["layernorm_fwd"] = orig_layernorm_fwd
                
                def patched_layernorm_fwd(*args, **kwargs):
                    # Check for fake tensors
                    has_fake = any(
                        isinstance(arg, FakeTensor)
                        for arg in args
                        if isinstance(arg, torch.Tensor)
                    )
                    
                    has_fake = has_fake or any(
                        isinstance(arg, FakeTensor)
                        for arg in kwargs.values()
                        if isinstance(arg, torch.Tensor)
                    )
                    
                    if has_fake:
                        print_rank_0("Handling fake tensor in layernorm_fwd")
                        # Get input tensor
                        input_tensor = None
                        for arg in args:
                            if isinstance(arg, torch.Tensor):
                                input_tensor = arg
                                break
                        
                        if input_tensor is not None:
                            # Create fake output and stats tensors
                            output = input_tensor.clone()
                            # Create fake mean and rstdev tensors with appropriate shapes
                            if len(input_tensor.shape) > 1:
                                reduced_shape = list(input_tensor.shape[:-1]) + [1]
                                means = torch.zeros(reduced_shape, 
                                                  dtype=input_tensor.dtype, 
                                                  device=input_tensor.device)
                                rstdevs = torch.ones(reduced_shape, 
                                                   dtype=input_tensor.dtype, 
                                                   device=input_tensor.device)
                            else:
                                means = torch.zeros(1, dtype=input_tensor.dtype, device=input_tensor.device)
                                rstdevs = torch.ones(1, dtype=input_tensor.dtype, device=input_tensor.device)
                            
                            return output, means, rstdevs
                        
                        return None, None, None
                    
                    return orig_layernorm_fwd(*args, **kwargs)
                
                transformer_engine.pytorch.ops.basic.layer_norm.layernorm_fwd = patched_layernorm_fwd
            
        except ImportError:
            print_rank_0("transformer_engine not found, skipping patching")
        except Exception as e:
            print_rank_0(f"Error patching transformer_engine: {e}")
        
        self.fake_mode.__enter__()
        self.memory_dispatch_mode.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager, disabling both fake tensors and memory estimation."""
        # Restore original distributed functions
        if hasattr(self, '_original_dist_funcs'):
            for func_name, orig_func in self._original_dist_funcs.items():
                setattr(torch.distributed, func_name, orig_func)
        
        # Restore original transformer_engine functions
        if hasattr(self, '_original_te_funcs'):
            try:
                import transformer_engine
                import transformer_engine.pytorch.ops as te_ops
                
                # Restore each patched function
                for func_path, orig_func in self._original_te_funcs.items():
                    if "." in func_path:
                        module_name, func_name = func_path.split(".")
                        if hasattr(te_ops, module_name):
                            module = getattr(te_ops, module_name)
                            setattr(module, func_name, orig_func)
                    elif func_path == "layernorm_fwd":
                        transformer_engine.pytorch.ops.basic.layer_norm.layernorm_fwd = orig_func
            except ImportError:
                pass
            except Exception as e:
                print_rank_0(f"Error restoring transformer_engine functions: {e}")
        
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
            if isinstance(device_id, int) or device_id.isdigit():
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
