import torch
from torch.utils._pytree import tree_map_only
import weakref
import functools
import contextlib
from collections import defaultdict
from typing import Optional
import traceback
import json
from .logger import get_logger, log_memory_table
from .mod_tracker import ModTracker

# Get a logger for this module
logger = get_logger(__name__)

class ProblematicOpsDispatchMode(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self, log_level=None):
        super().__init__()
        if log_level is not None:
            # Create a unique logger name for this instance
            self.logger = get_logger(f"{__name__}.ProblematicOpsDispatchMode.instance_{id(self)}")
            self.logger.setLevel(log_level)
        else:
            # Use the module-level logger by default, which is 'logger' from the outer scope of this file
            self.logger = logger

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs is not None else {}
        func_name = str(func)

        # List of problematic operations that need special handling
        problematic_ops_list = [
            "_local_scalar_dense",  # Data-dependent scalar conversion
            "random_",              # Random generation
            "item",                 # Tensor to scalar conversion
        ]

        if any(op_name in func_name for op_name in problematic_ops_list):
            # For random operations
            if "random_" in func_name:
                self.logger.debug(f"ProblematicOpsDispatchMode: Handling '{func_name}' by returning input tensor.")
                return args[0]  # Return the input tensor unchanged
            # For scalar conversion operations
            elif "_local_scalar_dense" in func_name or "item" in func_name:
                self.logger.debug(f"ProblematicOpsDispatchMode: Handling '{func_name}' by returning scalar 42.")
                return 42  # Return a fixed scalar value
            
            # Fallback for other ops in problematic_ops_list if not covered above
            try:
                self.logger.debug(f"ProblematicOpsDispatchMode: For '{func_name}', attempting to run with fake tensor mode disabled.")
                with torch._subclasses.fake_tensor._disable_fake_tensor_mode():
                    out = func(*args, **kwargs)
                self.logger.debug(f"ProblematicOpsDispatchMode: Successfully ran '{func_name}' with fake tensor mode disabled.")
                return out
            except Exception as e:
                self.logger.warning(
                    f"ProblematicOpsDispatchMode: Exception while running '{func_name}' with fake tensor mode disabled: {e}. "
                    "Falling back."
                )
                if len(args) > 0 and isinstance(args[0], torch.Tensor):
                    self.logger.debug(f"ProblematicOpsDispatchMode: Fallback for '{func_name}': returning input tensor.")
                    return args[0]
                self.logger.debug(f"ProblematicOpsDispatchMode: Fallback for '{func_name}': returning None.")
                return None
        
        # If func_name is not in problematic_ops_list, let the dispatch chain continue.

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
                if isinstance(split_sizes, torch._subclasses.fake_tensor.FakeTensor):
                    num_chunks = len(split_sizes)
                    self.logger.debug(f"Patching aten.split_with_sizes for FakeTensor. Using torch.chunk with {num_chunks} chunks along dim {dim}.")
                    # Use torch.chunk logic for uniform splitting
                    return torch.chunk(input_tensor, num_chunks, dim=dim)
                else:
                    return torch.split(input_tensor, split_sizes, dim=dim)

        return func(*args, **kwargs)

class MemoryDispatchMode(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self, log_level=None, mod_tracker: Optional[ModTracker] = None):
        self.peak_memory_per_device = {}
        self.current_memory_per_device = {}
        self.peak_memory_snapshot_per_device = {} # Stores snapshots {device_id: [info_dict1, info_dict2, ...]}
        # Stores {storage: {'size': nbytes, 'shape': shape, 'module': module_path}}
        self.live_tensors = weakref.WeakKeyDictionary()

        # New: track release stack traces for peak memory snapshot tensors
        self.peak_memory_tensor_release_info = {} # Stores {tensor_id: {'release_stack_trace': trace}}

        self.module_memory_usage = defaultdict(lambda: defaultdict(int))

        self.phase_tensor_modules = defaultdict(list)
        self.current_phase = "initialization"
        
        self.mod_tracker = mod_tracker
        self._internal_module_stack = ["Unknown"]

        if self.mod_tracker:
            self.mod_tracker.register_user_hooks(
                pre_fw_hook=self._user_pre_fw_hook,
                post_fw_hook=self._user_post_fw_hook
            )
        
        # Configure specific log level for this instance if provided
        if log_level is not None:
            self.logger = get_logger(f"{__name__}.instance_{id(self)}")
            self.logger.setLevel(log_level)
        else:
            self.logger = logger

    def _user_pre_fw_hook(self, module, _input):
        if self.mod_tracker:
            fqn = self.mod_tracker.get_known_fqn(module)
            if fqn:
                self._internal_module_stack.append(fqn)
            else:
                # Fallback if FQN is not found, though ModTracker should provide it
                self._internal_module_stack.append(f"UnidentifiedModule_{type(module).__name__}")
        else:
            # Should not happen if tracer passes mod_tracker
            self._internal_module_stack.append(f"NoModTracker_{type(module).__name__}")

    def _user_post_fw_hook(self, module, _input, _output):
        if len(self._internal_module_stack) > 1: # Keep the base "Unknown"
            # Pop the FQN associated with the completed module
            # We assume user hooks are called in a stack-like manner for nested modules
            self._internal_module_stack.pop()
        else:
            self.logger.warning(f"Attempted to pop from module stack for {module} but stack is too small or mismatched.")

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs is not None else {}

        func_name = str(func)

        current_module = self.current_module_path
        try:
            outputs = func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in {func_name}: {str(e)}")
            self.logger.error(f"{func_name}, {args}")
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

        # Capture stack trace
        current_stack_trace = traceback.format_stack()

        # Calculate memory usage
        nbytes = storage.nbytes()

        # Update current device memory usage
        if device_id not in self.current_memory_per_device:
            self.current_memory_per_device[device_id] = 0
            self.peak_memory_per_device[device_id] = 0
            self.peak_memory_snapshot_per_device[device_id] = [] # Initialize as list for new device

        self.current_memory_per_device[device_id] += nbytes
        
        # Generate a unique ID for this tensor
        tensor_id = id(storage)
        
        # Check if current memory exceeds peak memory for this device
        if self.current_memory_per_device[device_id] > self.peak_memory_per_device[device_id]:
            self.peak_memory_per_device[device_id] = self.current_memory_per_device[device_id]
            # Save a snapshot of live tensors' info for this device
            current_snapshot_list = []
            # Iterate over a copy of items from live_tensors for safety
            for _storage, info in list(self.live_tensors.items()):
                if info.get("device") == device_id: # Only tensors on the current device
                    # Create a copy of the info dict to avoid modifying the live_tensors entry
                    info_copy = info.copy()
                    info_copy["tensor_id"] = id(_storage)  # Add unique ID for later matching
                    current_snapshot_list.append(info_copy) 
            self.peak_memory_snapshot_per_device[device_id] = current_snapshot_list

        # Update module memory usage statistics
        self.module_memory_usage[module_path][device_id] += nbytes

        # Store tensor info keyed by storage
        self.live_tensors[storage] = {
            "size": nbytes,
            "shape": tensor.shape,
            "module": module_path,
            "device": device_id, # Store device for release
            "phase": self.current_phase, # Store creation phase
            "stack_trace": current_stack_trace, # Store stack trace
            "tensor_id": tensor_id  # Add unique ID
        }

        # # --- Patch storage.resize_ to intercept manual release --- 
        storage.resize_ = self.create_patched_resize_(nbytes, device_id)

        # Apply the patch - monkey patching the storage instance
        # This assumes storage objects are mutable and allow attribute assignment
        # storage.resize_ = patched_resize_
        # --- End Patch ---
        weakref.finalize(
            storage,
            functools.partial(self.release_memory, device_id, nbytes, tensor_id),
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

    @property
    def current_module_path(self):
        """Get the current module path from the internal stack."""
        if not self._internal_module_stack: # Should always have "Unknown" at least
            return "Unknown"
        return self._internal_module_stack[-1]

    def release_memory(self, device_id, nbytes, tensor_id=None):
        """Helper method to decrease memory count - called by finalize or patched resize_(0)."""
        # Capture stack trace at release time
        release_stack_trace = traceback.format_stack()
        
        if tensor_id is not None:
            # Store the release stack trace for this tensor
            self.peak_memory_tensor_release_info[tensor_id] = {
                "release_stack_trace": release_stack_trace
            }
            
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

    def log_peak_memory_snapshot(self, device_id=None, min_memory_mb=0):
        """Logs a table of tensors from the peak memory snapshot for specified device(s).

        Args:
            device_id (optional): Specific device (e.g., torch.device('cuda:0')) to log the snapshot for.
                                  If None, logs snapshots for all devices that have one.
            min_memory_mb (float): Minimum memory size (in MB) for a tensor to be included in the report.
                                   Defaults to 0 (show all).
        """
        devices_to_log = []
        if device_id is not None:
            if device_id in self.peak_memory_snapshot_per_device:
                devices_to_log.append(device_id)
            else:
                self.logger.info(f"No peak memory snapshot found for device {device_id}.")
                return
        else:
            devices_to_log = list(self.peak_memory_snapshot_per_device.keys())

        if not devices_to_log:
            self.logger.info("No peak memory snapshots available to log.")
            return

        min_memory_bytes = min_memory_mb * (1024**2)

        for dev_id in devices_to_log:
            snapshot_tensors_info = []
            total_snapshot_memory = 0
            
            device_snapshot_list = self.peak_memory_snapshot_per_device.get(dev_id, [])
            if not device_snapshot_list:
                self.logger.info(f"Peak memory snapshot for device {dev_id} is empty or not found.")
                continue

            # Iterate through the snapshot items (list of info dicts)
            for info in device_snapshot_list:
                # We assume info dict contains all necessary details like 'size', 'shape', 'module', 'phase', 'stack_trace'
                snapshot_tensors_info.append({
                    "size": info["size"],
                    "shape": info["shape"],
                    "module": info["module"],
                    "phase": info.get("phase", "N/A"), 
                    "stack_trace": info.get("stack_trace", []) # Ensure stack_trace is handled
                })
                total_snapshot_memory += info["size"]

            # Filter by minimum memory size
            filtered_snapshot_info = [
                info for info in snapshot_tensors_info if info["size"] >= min_memory_bytes
            ]

            # Sort by size descending
            filtered_snapshot_info.sort(key=lambda x: x["size"], reverse=True)

            # Format and log the table
            rows = []
            for info in filtered_snapshot_info:
                module_str = ".".join(info['module'].split('.')[-5:])
                phase_str = info['phase'][:18] + '..' if len(info['phase']) > 20 else info['phase']
                size_mb = info['size'] / (1024**2)
                shape_str = str(info['shape'])
                rows.append([phase_str, module_str, f"{size_mb:.2f}", shape_str])
            
            log_title = f"Peak Memory Snapshot for Device {dev_id} (Min Size: {min_memory_mb} MB, Peak: {self.peak_memory_per_device.get(dev_id, 0)/(1024**2):.2f} MB):"
            log_memory_table(
                self.logger,
                log_title,
                ["Phase", "Module", "Size (MB)", "Shape"],
                rows
            )
            
            total_snapshot_mb = total_snapshot_memory / (1024**2)
            self.logger.info(f"Total Memory in Snapshot for {dev_id}: {total_snapshot_mb:.2f} MB")

    def save_peak_memory_snapshot_to_file(self, filepath: str, device_id=None, min_memory_mb=0):
        """Saves the peak memory snapshot to a JSON file.

        Args:
            filepath (str): Path to the file where the snapshot will be saved.
            device_id (optional): Specific device (e.g., torch.device('cuda:0')) to save the snapshot for.
                                  If None, saves snapshots for all devices that have one.
            min_memory_mb (float): Minimum memory size (in MB) for a tensor to be included in the snapshot.
                                   Defaults to 0 (show all).
        """
        devices_to_log = []
        if device_id is not None:
            # Convert device_id to string if it's a torch.device object for consistent key access
            device_key = str(device_id) if isinstance(device_id, torch.device) else device_id
            if device_key in self.peak_memory_snapshot_per_device:
                devices_to_log.append(device_key)
            elif device_id in self.peak_memory_snapshot_per_device: # Fallback for direct device object
                devices_to_log.append(device_id)
            else:
                self.logger.info(f"No peak memory snapshot found for device {device_id}.")
                return
        else:
            devices_to_log = list(self.peak_memory_snapshot_per_device.keys())

        if not devices_to_log:
            self.logger.info("No peak memory snapshots available to save.")
            return

        min_memory_bytes = min_memory_mb * (1024**2)
        snapshot_data_to_save = {}

        for dev_id_key in devices_to_log:
            # Use dev_id_key which is consistently a string if conversion happened
            actual_dev_id_for_lookup = dev_id_key
            # If original keys are torch.device objects, we might need to find the correct one
            if not isinstance(dev_id_key, (str)) and dev_id_key not in self.peak_memory_snapshot_per_device:
                 # Attempt to find by string representation if original keys are torch.device
                 found = False
                 for k_dev in self.peak_memory_snapshot_per_device.keys():
                     if str(k_dev) == str(dev_id_key):
                         actual_dev_id_for_lookup = k_dev
                         found = True
                         break
                 if not found:
                     self.logger.warning(f"Could not reliably map key {dev_id_key} for snapshot saving.")
                     continue


            snapshot_tensors_info = []
            
            device_snapshot_list = self.peak_memory_snapshot_per_device.get(actual_dev_id_for_lookup, [])
            if not device_snapshot_list:
                self.logger.info(f"Peak memory snapshot for device {actual_dev_id_for_lookup} is empty or not found during save.")
                continue

            for info_item in device_snapshot_list:
                if info_item["size"] >= min_memory_bytes:
                    # Get release stack trace if available
                    tensor_id = info_item.get("tensor_id")
                    release_info = {}
                    if tensor_id is not None and tensor_id in self.peak_memory_tensor_release_info:
                        release_info = self.peak_memory_tensor_release_info[tensor_id]
                    
                    # Prepare info for JSON serialization
                    serializable_info = {
                        "size_bytes": info_item["size"],
                        "size_mb": info_item["size"] / (1024**2),
                        "shape": str(info_item["shape"]), # Convert torch.Size to string
                        "module": info_item["module"],
                        "phase": info_item.get("phase", "N/A"),
                        "create_stack_trace": info_item.get("stack_trace", []), # Include creation stack trace
                        "release_stack_trace": release_info.get("release_stack_trace", []) # Include release stack trace if available
                    }
                    snapshot_tensors_info.append(serializable_info)
            
            # Sort by size descending before saving
            snapshot_tensors_info.sort(key=lambda x: x["size_bytes"], reverse=True)
            
            # Use string representation of device ID as key in the output file
            snapshot_data_to_save[str(actual_dev_id_for_lookup)] = {
                "peak_memory_mb": self.peak_memory_per_device.get(actual_dev_id_for_lookup, 0) / (1024**2),
                "tensors": snapshot_tensors_info
            }

        try:
            with open(filepath, 'w') as f:
                json.dump(snapshot_data_to_save, f, indent=4)
            self.logger.info(f"Peak memory snapshot saved to {filepath}")
        except IOError as e:
            self.logger.error(f"Failed to save peak memory snapshot to {filepath}: {e}")
        except TypeError as e:
            self.logger.error(f"TypeError during JSON serialization for {filepath}: {e}. Ensure all data is serializable.")

    @contextlib.contextmanager
    def trace_module(self, name: str):
        """Context manager for manually tracking non-module code blocks"""
        self._internal_module_stack.append(name)
        try:
            yield
        finally:
            if self._internal_module_stack and self._internal_module_stack[-1] == name:
                self._internal_module_stack.pop()
            else:
                self.logger.warning(
                    f"Mismatched pop in trace_module for '{name}'. Current stack top: "
                    f"'{self._internal_module_stack[-1] if self._internal_module_stack else 'EMPTY'}'. "
                    f"Full stack: {self._internal_module_stack}"
                ) 