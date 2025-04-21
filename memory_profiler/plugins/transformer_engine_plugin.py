import torch
import functools
from .base_plugin import TracerPlugin
from ..utils import print_rank_0

class TransformerEnginePlugin(TracerPlugin):
    """Plugin for handling transformer_engine operations with fake tensors."""
    
    def setup(self, tracer):
        self.tracer = tracer
        self.original_te_funcs = {}
        self._tex_module = None # Store the tex module reference
    
    def _create_patched_norm_func(self, func_name, orig_func):
        """Creates a patched version of a normalization function to handle fake tensors."""
        from torch._subclasses.fake_tensor import FakeTensor

        @functools.wraps(orig_func)
        def patched_func(*args, **kwargs):
            # Check for fake tensors in args and kwargs
            has_fake = any(isinstance(arg, FakeTensor) for arg in args if isinstance(arg, torch.Tensor)) or \
                       any(isinstance(v, FakeTensor) for v in kwargs.values() if isinstance(v, torch.Tensor))

            if has_fake:
                print_rank_0(f"Handling fake tensor in {func_name}")
                # Identify the main input tensor (usually the first tensor arg)
                input_tensor = None
                for arg in args:
                    if isinstance(arg, FakeTensor):
                        input_tensor = arg
                        break
                if input_tensor is None:
                    for v in kwargs.values():
                        if isinstance(v, FakeTensor):
                            input_tensor = v
                            break

                if input_tensor is None:
                    print_rank_0(f"Warning: Fake tensor detected in {func_name} but couldn't identify input tensor. Falling back to original function.")
                    return orig_func(*args, **kwargs)

                # --- Create fake outputs based on func_name ---
                if func_name in ["layernorm_fwd", "rmsnorm_fwd"]:
                    # Output: output, mu/None, rsigma
                    output = input_tensor.clone()
                    stats_dtype = torch.float32 # Common practice for stats
                    if len(input_tensor.shape) > 1:
                        reduced_shape = list(input_tensor.shape[:-1]) + [1]
                        means = torch.zeros(reduced_shape, dtype=stats_dtype, device=input_tensor.device)
                        rstdevs = torch.ones(reduced_shape, dtype=stats_dtype, device=input_tensor.device)
                    else:
                        means = torch.zeros(1, dtype=stats_dtype, device=input_tensor.device)
                        rstdevs = torch.ones(1, dtype=stats_dtype, device=input_tensor.device)

                    if func_name == "layernorm_fwd":
                        return output, means, rstdevs
                    else: # rmsnorm_fwd
                        return output, None, rstdevs

                elif func_name == "layernorm_bwd":
                    # Args typically: grad_output, input, mu, rsigma, ln_weight, ...
                    grad_output_tensor = args[0] if len(args) > 0 and isinstance(args[0], FakeTensor) else None
                    ln_weight_tensor = args[4] if len(args) > 4 and isinstance(args[4], FakeTensor) else None

                    if grad_output_tensor is None or ln_weight_tensor is None:
                        print_rank_0(f"Warning: Couldn't identify grad_output or ln_weight for fake {func_name}. Using input tensor shape for grads.")
                        dgrad = input_tensor.clone() # Best guess shape
                        # Guess weight shape based on last dim of input
                        dgamma = torch.zeros(input_tensor.shape[-1], dtype=input_tensor.dtype, device=input_tensor.device)
                        dbeta = torch.zeros(input_tensor.shape[-1], dtype=input_tensor.dtype, device=input_tensor.device)
                        return dgrad, dgamma, dbeta

                    dgrad = grad_output_tensor.clone()
                    dgamma = torch.zeros_like(ln_weight_tensor)
                    dbeta = torch.zeros_like(ln_weight_tensor)
                    return dgrad, dgamma, dbeta

                elif func_name == "rmsnorm_bwd":
                    # Args typically: grad_output, input, rsigma, ln_weight, ...
                    grad_output_tensor = args[0] if len(args) > 0 and isinstance(args[0], FakeTensor) else None
                    ln_weight_tensor = args[3] if len(args) > 3 and isinstance(args[3], FakeTensor) else None

                    if grad_output_tensor is None or ln_weight_tensor is None:
                         print_rank_0(f"Warning: Couldn't identify grad_output or ln_weight for fake {func_name}. Using input tensor shape for grads.")
                         dgrad = input_tensor.clone() # Best guess shape
                         # Guess weight shape based on last dim of input
                         dgamma = torch.zeros(input_tensor.shape[-1], dtype=input_tensor.dtype, device=input_tensor.device)
                         return dgrad, dgamma

                    dgrad = grad_output_tensor.clone()
                    dgamma = torch.zeros_like(ln_weight_tensor)
                    return dgrad, dgamma
                else:
                    print_rank_0(f"Error: Unhandled function name {func_name} in patch. Calling original.")
                    return orig_func(*args, **kwargs) # Fallback

            else:
                # No fake tensor, call original function
                return orig_func(*args, **kwargs)

        return patched_func

    def enter(self):
        try:
            # Import tex here to ensure it's available
            import transformer_engine.pytorch.cpp_extensions as tex
            self._tex_module = tex # Store for restoration
            from torch._subclasses.fake_tensor import FakeTensor # Import here for use in _create_patched_norm_func


            def custom_prepare_for_saving(self):
                return self, None
            FakeTensor.prepare_for_saving = custom_prepare_for_saving

            # Functions to patch in tex module
            funcs_to_patch = [
                "layernorm_fwd",
                "rmsnorm_fwd",
                "layernorm_bwd",
                "rmsnorm_bwd",
            ]

            for func_name in funcs_to_patch:
                if hasattr(tex, func_name):
                    original_func = getattr(tex, func_name)
                    # Avoid double-patching if enter is called again somehow
                    if func_name not in self.original_te_funcs:
                        self.original_te_funcs[func_name] = original_func
                        patched_func = self._create_patched_norm_func(func_name, original_func)
                        setattr(tex, func_name, patched_func)
                        print_rank_0(f"Patched transformer_engine.pytorch.cpp_extensions.{func_name} for fake tensors")
                else:
                    print_rank_0(f"Function {func_name} not found in transformer_engine.pytorch.cpp_extensions, skipping patch.")

            # Clean up any potential old patches from previous versions of this code
            # (assuming old version patched transformer_engine.pytorch.ops.basic.layer_norm)
            if "layernorm_fwd" in self.original_te_funcs:
                 try:
                     import transformer_engine.pytorch.ops.basic.layer_norm as layer_norm_ops
                     if hasattr(layer_norm_ops, 'layernorm_fwd'):
                         # If the function in ops is different from the tex one we stored,
                         # it might be an old patch; restore the tex one there too for safety.
                         if layer_norm_ops.layernorm_fwd is not self.original_te_funcs["layernorm_fwd"]:
                              # Check if it's actually the *patched* function we just created
                              # This check is tricky, maybe just ensure the tex one is set
                              pass # Let the new setattr(tex, ...) dominate
                 except ImportError:
                     pass # Module doesn't exist, nothing to clean

        except ImportError:
            print_rank_0("transformer_engine.pytorch.cpp_extensions not found, skipping patching.")
            self._tex_module = None
        except Exception as e:
            print_rank_0(f"Error patching transformer_engine functions: {e}")
            import traceback
            traceback.print_exc()

    def exit(self, exc_type, exc_val, exc_tb):
        # Restore original transformer_engine functions
        if self._tex_module is not None:
            try:
                for func_name, orig_func in self.original_te_funcs.items():
                    # Directly restore onto the stored tex module
                    if hasattr(self._tex_module, func_name):
                        current_func = getattr(self._tex_module, func_name)
                        # Restore only if it's different from original (i.e., it's patched)
                        if current_func is not orig_func:
                             setattr(self._tex_module, func_name, orig_func)
                             print_rank_0(f"Restored transformer_engine.pytorch.cpp_extensions.{func_name}")
                    else:
                        print_rank_0(f"Warning: Could not find {func_name} during restoration.")

            except Exception as e:
                print_rank_0(f"Error restoring transformer_engine functions: {e}")
                import traceback
                traceback.print_exc()
        else:
            print_rank_0("No tex module reference found, skipping restoration.")

        # Clear the stored functions and module reference
        self.original_te_funcs.clear()
        self._tex_module = None 
