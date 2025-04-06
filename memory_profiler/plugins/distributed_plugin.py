import torch
import functools
from .base_plugin import TracerPlugin
from ..utils import print_rank_0

class DistributedPlugin(TracerPlugin):
    """Plugin for handling distributed operations with fake tensors."""
    
    def setup(self, tracer):
        self.tracer = tracer
        self.original_funcs = {}
    
    def enter(self):
        if not hasattr(torch, 'distributed') or not torch.distributed.is_available():
            return
        
        # Import FakeTensor for proper type checking
        from torch._subclasses.fake_tensor import FakeTensor
        
        # Get all distributed functions that might need patching
        dist_functions = [name for name in dir(torch.distributed) 
                         if callable(getattr(torch.distributed, name)) 
                         and not name.startswith('_')]
        
        # Create patches for each distributed function
        for func_name in dist_functions:
            original_func = getattr(torch.distributed, func_name)
            self.original_funcs[func_name] = original_func
            
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
    
    def exit(self, exc_type, exc_val, exc_tb):
        # Restore original distributed functions
        for func_name, orig_func in self.original_funcs.items():
            setattr(torch.distributed, func_name, orig_func) 