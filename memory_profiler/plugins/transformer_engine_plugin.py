import torch
import functools
from .base_plugin import TracerPlugin
from ..utils import print_rank_0

class TransformerEnginePlugin(TracerPlugin):
    """Plugin for handling transformer_engine operations with fake tensors."""
    
    def setup(self, tracer):
        self.tracer = tracer
        self.original_te_funcs = {}
    
    def enter(self):
        try:
            import transformer_engine
            import transformer_engine.pytorch.ops as te_ops
            from torch._subclasses.fake_tensor import FakeTensor
            
            # Patch forward methods on transformer_engine module ops
            for op_name in dir(te_ops):
                op_obj = getattr(te_ops, op_name)
                if hasattr(op_obj, 'forward') and callable(getattr(op_obj, 'forward')):
                    try:
                        orig_forward = getattr(op_obj, 'forward')
                        self.original_te_funcs[f"{op_name}.forward"] = orig_forward
                        
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
                self.original_te_funcs["layernorm_fwd"] = orig_layernorm_fwd
                
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
    
    def exit(self, exc_type, exc_val, exc_tb):
        # Restore original transformer_engine functions
        try:
            import transformer_engine
            import transformer_engine.pytorch.ops as te_ops
            
            # Restore each patched function
            for func_path, orig_func in self.original_te_funcs.items():
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