import torch
import functools
from .base_plugin import TracerPlugin
from ..core.logger import get_logger

# Get a logger for this module
logger = get_logger(__name__)

class DistributedPlugin(TracerPlugin):
    """Plugin for handling distributed operations with fake tensors."""

    def setup(self, tracer):
        self.tracer = tracer
        self.original_funcs = {}
        self.logger = get_logger(f"{__name__}.instance_{id(self)}")

    def enter(self):
        if not hasattr(torch, "distributed") or not torch.distributed.is_available():
            return

        # Import FakeTensor for proper type checking
        from torch._subclasses.fake_tensor import FakeTensor

        # Get all distributed functions that might need patching
        # dist_functions = [
        #     name
        #     for name in dir(torch.distributed)
        #     if callable(getattr(torch.distributed, name)) and not name.startswith("_")
        # ]
        dist_functions = [
            "all_gather",
            "all_gather_coalesced",
            "all_gather_into_tensor",
            "all_gather_object",
            "all_reduce",
            "all_reduce_coalesced",
            "all_to_all",
            "all_to_all_single",
            "barrier",
            "batch_isend_irecv",
            "breakpoint",
            "broadcast",
            "broadcast_object_list",
            "gather",
            "gather_object",
            "irecv",
            "isend",
            "recv",
            "recv_object_list",
            "reduce",
            "reduce_scatter",
            "reduce_scatter_tensor",
            "scatter",
            "scatter_object_list",
            "send",
            "send_object_list",
            "_reduce_scatter_base",
            "_all_gather_base",
        ]
        
        # Define a dummy class that supports attribute assignment
        class DummyDistributedOutput:
            def wait(self):
                pass
                
        # Create patches for each distributed function
        for func_name in dist_functions:
            original_func = getattr(torch.distributed, func_name)
            self.original_funcs[func_name] = original_func

            # Define the patched function
            def make_patched_dist_func(orig_func):
                @functools.wraps(orig_func)
                def patched_dist_func(*args, **kwargs):
                    # self.logger.debug(f"Calling {orig_func.__name__} with args: {args} and kwargs: {kwargs}")
                    return DummyDistributedOutput()

                return patched_dist_func
            self.logger.debug(f"Setting {func_name} to patched function")
            setattr(torch.distributed, func_name, make_patched_dist_func(original_func))

    def exit(self, exc_type, exc_val, exc_tb):
        # Restore original distributed functions
        for func_name, orig_func in self.original_funcs.items():
            setattr(torch.distributed, func_name, orig_func)
            self.logger.debug(f"Restored original {func_name} function")
