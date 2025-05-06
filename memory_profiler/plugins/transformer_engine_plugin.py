from typing import Optional, Tuple
import torch
import functools
from .base_plugin import TracerPlugin
from ..core.logger import get_logger
import importlib

# Get a logger for this module
logger = get_logger(__name__)

def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x

def round_multiple(x, m):
    return (x + m - 1) // m * m

class TransformerEnginePlugin(TracerPlugin):
    """Plugin for handling transformer_engine operations with fake tensors."""
    
    def setup(self, tracer):
        self.tracer = tracer
        self.original_te_funcs = {}
        self._tex_module = None # Store the tex module reference
        self._pytorch_tex_module = None # Store the gemm module reference
        self.logger = get_logger(f"{__name__}.instance_{id(self)}")
    
    def _create_patched_norm_func(self, func_name, orig_func):
        """Creates a patched version of a normalization function to handle fake tensors."""
        from torch._subclasses.fake_tensor import FakeTensor

        @functools.wraps(orig_func)
        def patched_func(*args, **kwargs):
            # Check for fake tensors in args and kwargs
            has_fake = any(isinstance(arg, FakeTensor) for arg in args if isinstance(arg, torch.Tensor)) or \
                       any(isinstance(v, FakeTensor) for v in kwargs.values() if isinstance(v, torch.Tensor))

            if has_fake:
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
                    self.logger.warning(f"Fake tensor detected in {func_name} but couldn't identify input tensor. Falling back to original function.")
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
                        self.logger.warning(f"Couldn't identify grad_output or ln_weight for fake {func_name}. Using input tensor shape for grads.")
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
                         self.logger.warning(f"Couldn't identify grad_output or ln_weight for fake {func_name}. Using input tensor shape for grads.")
                         dgrad = input_tensor.clone() # Best guess shape
                         # Guess weight shape based on last dim of input
                         dgamma = torch.zeros(input_tensor.shape[-1], dtype=input_tensor.dtype, device=input_tensor.device)
                         return dgrad, dgamma

                    dgrad = grad_output_tensor.clone()
                    dgamma = torch.zeros_like(ln_weight_tensor)
                    return dgrad, dgamma
                else:
                    self.logger.error(f"Unhandled function name {func_name} in patch. Calling original.")
                    return orig_func(*args, **kwargs) # Fallback

            else:
                # No fake tensor, call original function
                return orig_func(*args, **kwargs)

        return patched_func

    def _create_patched_gemm_func(self, func_name, orig_func):
        """Creates a patched version of a GEMM function to handle fake tensors."""
        from torch._subclasses.fake_tensor import FakeTensor

        @functools.wraps(orig_func)
        def patched_func(*args, **kwargs):
            A, trans_A, B, trans_B = args[:4]
            # Check for fake tensors in A and B (usually first two tensor args)
            has_fake = isinstance(A, FakeTensor) or isinstance(B, FakeTensor)

            if has_fake:
                # general_gemm returns: out, bias_grad, gelu_input, extra_output
                # Return None placeholders for now as requested
                A = torch.squeeze(A)
                A = A.reshape(-1, A.shape[-1])
                B = torch.squeeze(B)
                if trans_A:
                    A = A.transpose(0, 1)
                if trans_B:
                    B = B.transpose(0, 1)
                out = B @ A

                return out, None, None, None
            else:
                # No fake tensor, call original function
                return orig_func(*args, **kwargs)

        return patched_func
    
    def _create_patched_grouped_gemm_func(self, func_name, orig_func):
        """Creates a patched version of a GEMM function to handle fake tensors."""
        from torch._subclasses.fake_tensor import FakeTensor

        @functools.wraps(orig_func)
        def patched_func(*args, **kwargs):
            A, trans_A, B, trans_B = args[:4]
            # import IPython; IPython.embed(); exit(0)
            out = args[4]
            # Check for fake tensors in A and B (usually first two tensor args)
            has_fake = isinstance(A[0], FakeTensor) or isinstance(B[0], FakeTensor)

            if has_fake:
                # general_gemm returns: out, bias_grad, gelu_input, extra_output
                # Return None placeholders for now as requested
                for i in range(len(A)):
                    if trans_A:
                        a = A[i].transpose(0, 1)
                    else:
                        a = A[i]
                    if trans_B:
                        b = B[i].transpose(0, 1)
                    else:
                        b = B[i]
                    _ = b @ a
                return [None] * len(A)
            else:
                # No fake tensor, call original function
                return orig_func(*args, **kwargs)

        return patched_func

    def _patch_flash_attn_fake(self):
        @torch.library.register_fake("flash_attn::_flash_attn_forward")
        def _flash_attn_forward_fake(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            dropout_p: float,
            softmax_scale: float,
            causal: bool,
            window_size_left: int,
            window_size_right: int,
            softcap: float,
            alibi_slopes: Optional[torch.Tensor],
            return_softmax: bool
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
            batch_size, seqlen_q, num_q_heads, q_head_size = q.shape
            batch_size, seqlen_q, num_v_heads, v_head_size = v.shape
            seqlen_k = k.shape[1]
            out = torch.empty((batch_size, seqlen_q, num_q_heads, v_head_size), dtype=q.dtype, device=q.device, layout=q.layout)
            softmax_lse = torch.empty((batch_size, num_q_heads, seqlen_q), dtype=torch.float32, device=q.device, layout=q.layout)
            p = torch.empty((0,), dtype=q.dtype, device=q.device, layout=q.layout)
            if return_softmax:
                p = torch.empty((batch_size, num_q_heads, round_multiple(seqlen_q, 128), round_multiple(seqlen_k, 128)), dtype=q.dtype, device=q.device, layout=q.layout)
            rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)

            return out, softmax_lse, p, rng_state

        @torch.library.register_fake("flash_attn::_flash_attn_backward")
        def _flash_attn_backward_fake(
            dout: torch.Tensor,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            out: torch.Tensor,
            softmax_lse: torch.Tensor,
            dq: Optional[torch.Tensor],
            dk: Optional[torch.Tensor],
            dv: Optional[torch.Tensor],
            dropout_p: float,
            softmax_scale: float,
            causal: bool,
            window_size_left: int,
            window_size_right: int,
            softcap: float,
            alibi_slopes: Optional[torch.Tensor],
            deterministic: bool,
            rng_state: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
            batch_size, seqlen_q, num_heads, _ = q.shape
            softmax_d = torch.empty((batch_size, num_heads, round_multiple(seqlen_q, 128)), device=q.device, dtype=torch.float32)
            return softmax_d

        self._flash_attn_forward_fake = _flash_attn_forward_fake
        self._flash_attn_backward_fake = _flash_attn_backward_fake


    def _patch_flash_attn_func(self):
        module = importlib.import_module("transformer_engine.pytorch.attention")

        class PatchedFlashAttnFunc(torch.autograd.Function):
            @staticmethod
            def forward(
                ctx,
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                softcap,
                alibi_slopes,
                deterministic,
                return_softmax,
                is_grad_enabled,
            ):
                is_grad = is_grad_enabled and any(
                    x.requires_grad for x in [q, k, v]
                )
                if softmax_scale is None:
                    softmax_scale = q.shape[-1] ** (-0.5)
                head_size_og = q.size(3)
                if head_size_og % 8 != 0:
                    q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
                    k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
                    v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
                out_padded, softmax_lse, S_dmask, rng_state = self._flash_attn_forward_fake(
                    q,
                    k,
                    v,
                    dropout_p,
                    softmax_scale,
                    causal=causal,
                    window_size_left=window_size[0],
                    window_size_right=window_size[1],
                    softcap=softcap,
                    alibi_slopes=alibi_slopes,
                    return_softmax=return_softmax and dropout_p > 0,
                )
                if is_grad:
                    ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
                    ctx.dropout_p = dropout_p
                    ctx.softmax_scale = softmax_scale
                    ctx.causal = causal
                    ctx.window_size = window_size
                    ctx.softcap = softcap
                    ctx.alibi_slopes = alibi_slopes
                    ctx.deterministic = deterministic
                out = out_padded[..., :head_size_og]
                return out if not return_softmax else (out, softmax_lse, S_dmask)

            @staticmethod
            def backward(ctx, dout, *args):
                q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
                dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
                head_size_og = dout.size(3)
                dout_padded = dout
                if head_size_og % 8 != 0:
                    dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
                self._flash_attn_backward_fake(
                    dout_padded,
                    q,
                    k,
                    v,
                    out,
                    softmax_lse,
                    dq,
                    dk,
                    dv,
                    ctx.dropout_p,
                    ctx.softmax_scale,
                    ctx.causal,
                    ctx.window_size[0],
                    ctx.window_size[1],
                    ctx.softcap,
                    ctx.alibi_slopes,
                    ctx.deterministic,
                    rng_state=rng_state,
                )
                return dq, dk, dv, None, None, None, None, None, None, None, None, None

        origin_func = getattr(module, "flash_attn_func")

        @functools.wraps(origin_func)
        def patched_flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
            window_size=(-1, -1),  # -1 means infinite context window
            softcap=0.0, # 0.0 means deactivated
            alibi_slopes=None,
            deterministic=False,
            return_attn_probs=False,
        ):
            # import IPython; IPython.embed(); exit(0)
            return PatchedFlashAttnFunc.apply(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                softcap,
                alibi_slopes,
                deterministic,
                return_attn_probs,
                torch.is_grad_enabled(),
            )

        module.flash_attn_func = patched_flash_attn_func


    def _patch_te_attn_backend(self):
        module = importlib.import_module("transformer_engine.pytorch.dot_product_attention.utils")
        orig_func = getattr(module, "get_attention_backend")
        from packaging.version import Version as PkgVersion
        @functools.wraps(orig_func)
        def patched_func(*args, **kwargs):
            return (
                True,
                PkgVersion("2.7.3"),
                False,
                None,
                False,
                [True, False, False],
            )

        module.get_attention_backend = patched_func

    def enter(self):
        try:
            # Import tex here to ensure it's available
            import transformer_engine.pytorch.cpp_extensions as tex
            self._tex_module = tex # Store for restoration
            # Import gemm submodule
            try:
                 import transformer_engine_torch as pytorch_tex
                 self._pytorch_tex_module = pytorch_tex # Store for restoration
            except ImportError:
                 self.logger.warning("transformer_engine_torch not found, skipping gemm patching.")
                 self._pytorch_tex_module = None

            from torch._subclasses.fake_tensor import FakeTensor # Import here for use in patch funcs


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
                        self.logger.debug(f"Patched transformer_engine.pytorch.cpp_extensions.{func_name} for fake tensors")
                else:
                    self.logger.warning(f"Function {func_name} not found in transformer_engine.pytorch.cpp_extensions, skipping patch.")

            for func_name in funcs_to_patch:
                if hasattr(pytorch_tex, func_name):
                    original_func = getattr(pytorch_tex, func_name)
                    # Avoid double-patching if enter is called again somehow
                    patched_func = self._create_patched_norm_func(func_name, original_func)
                    setattr(pytorch_tex, func_name, patched_func)
                    self.logger.debug(f"Patched transformer_engine_torch.{func_name} for fake tensors")
                else:
                    self.logger.warning(f"Function {func_name} not found in transformer_engine_torch, skipping patch.")

            # Patch general_gemm in the gemm submodule
            gemm_func_name = "generic_gemm"
            if self._pytorch_tex_module and hasattr(self._pytorch_tex_module, gemm_func_name):
                 original_func = getattr(self._pytorch_tex_module, gemm_func_name)
                 if gemm_func_name not in self.original_te_funcs:
                     self.original_te_funcs[gemm_func_name] = original_func
                     patched_func = self._create_patched_gemm_func(gemm_func_name, original_func)
                     setattr(self._pytorch_tex_module, gemm_func_name, patched_func)
                     self.logger.debug(f"Patched transformer_engine.pytorch.cpp_extensions.gemm.{gemm_func_name} for fake tensors")
            elif self._pytorch_tex_module:
                 self.logger.warning(f"Function {gemm_func_name} not found in transformer_engine.pytorch.cpp_extensions.gemm, skipping patch.")

            # Patch grouped_gemm in the gemm submodule
            gemm_func_name = "te_general_grouped_gemm"
            if self._pytorch_tex_module and hasattr(self._pytorch_tex_module, gemm_func_name):
                 original_func = getattr(self._pytorch_tex_module, gemm_func_name)
                 if gemm_func_name not in self.original_te_funcs:
                     self.original_te_funcs[gemm_func_name] = original_func
                     patched_func = self._create_patched_grouped_gemm_func(gemm_func_name, original_func)
                     setattr(self._pytorch_tex_module, gemm_func_name, patched_func)
                     self.logger.debug(f"Patched transformer_engine.pytorch.cpp_extensions.gemm.{gemm_func_name} for fake tensors")
            elif self._pytorch_tex_module:
                 self.logger.warning(f"Function {gemm_func_name} not found in transformer_engine.pytorch.cpp_extensions.gemm, skipping patch.")

            # Functions to patch in rmsnorm module
            funcs_to_patch = [
                "rmsnorm_fwd",
                "rmsnorm_bwd",
            ]

            module = importlib.import_module("transformer_engine.pytorch.ops.basic.rmsnorm")
            for func_name in funcs_to_patch:
                if hasattr(module, func_name):
                    original_func = getattr(module, func_name)
                    patched_func = self._create_patched_norm_func(func_name, original_func)
                    setattr(module, func_name, patched_func)
                    self.logger.debug(f"Patched transformer_engine.pytorch.ops.basic.rmsnorm.{func_name} for fake tensors")
                else:
                    self.logger.warning(f"Function {func_name} not found in transformer_engine.pytorch.ops.basic.rmsnorm, skipping patch.")

            # Functions to patch in layernorm module
            funcs_to_patch = [
                "layernorm_fwd",
                "layernorm_bwd",
            ]

            module = importlib.import_module("transformer_engine.pytorch.ops.basic.layer_norm")
            for func_name in funcs_to_patch:
                if hasattr(module, func_name):
                    original_func = getattr(module, func_name)
                    patched_func = self._create_patched_norm_func(func_name, original_func)
                    setattr(module, func_name, patched_func)
                    self.logger.debug(f"Patched transformer_engine.pytorch.ops.basic.layer_norm.{func_name} for fake tensors")
                else:
                    self.logger.warning(f"Function {func_name} not found in transformer_engine.pytorch.ops.basic.layer_norm, skipping patch.")

            self._patch_te_attn_backend()
            self._patch_flash_attn_fake()
            self._patch_flash_attn_func()

        except ImportError:
            self.logger.warning("transformer_engine.pytorch.cpp_extensions not found, skipping patching.")
            self._tex_module = None
        except Exception as e:
            self.logger.error(f"Error patching transformer_engine functions: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def exit(self, exc_type, exc_val, exc_tb):
        restored_keys = set()
        # Restore original transformer_engine functions from tex module
        if self._tex_module is not None:
            try:
                for func_name, orig_func in self.original_te_funcs.items():
                    # Check if this function belongs to the tex module (heuristic: not gemm)
                    if func_name != "general_gemm" and hasattr(self._tex_module, func_name):
                        current_func = getattr(self._tex_module, func_name)
                        # Restore only if it's different from original (i.e., it's patched)
                        # Or if current_func is None (might happen with complex patching scenarios)
                        if current_func is not orig_func:
                             setattr(self._tex_module, func_name, orig_func)
                             self.logger.debug(f"Restored transformer_engine.pytorch.cpp_extensions.{func_name}")
                             restored_keys.add(func_name)
                    # elif func_name != "general_gemm": # Only print warning if tex module exists but func doesn't
                    #     self.logger.warning(f"Could not find {func_name} in tex during restoration.")

            except Exception as e:
                self.logger.error(f"Error restoring transformer_engine tex functions: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        # else:
            # self.logger.debug("No tex module reference found, skipping tex restoration.")

        # Restore original transformer_engine functions from gemm module
        if self._pytorch_tex_module is not None:
            try:
                for gemm_func_name in self.original_te_funcs:
                    orig_func = self.original_te_funcs[gemm_func_name]
                    if hasattr(self._pytorch_tex_module, gemm_func_name):
                        current_func = getattr(self._pytorch_tex_module, gemm_func_name)
                    if current_func is not orig_func:
                        setattr(self._pytorch_tex_module, gemm_func_name, orig_func)
                        self.logger.debug(f"Restored transformer_engine.pytorch.cpp_extensions.gemm.{gemm_func_name}")
                        restored_keys.add(gemm_func_name)
            except Exception as e:
                self.logger.error(f"Error restoring transformer_engine gemm functions: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        # else:
             # self.logger.debug("No gemm module reference found, skipping gemm restoration.")


        # Clear the stored functions and module references
        # Remove restored keys from the dictionary
        for key in restored_keys:
             if key in self.original_te_funcs:
                 del self.original_te_funcs[key]

        # Print warning for any remaining keys (shouldn't happen ideally)
        if self.original_te_funcs:
            self.logger.warning(f"Some original TE functions may not have been restored: {list(self.original_te_funcs.keys())}")
            self.original_te_funcs.clear() # Clear anyway

        self._tex_module = None
        self._pytorch_tex_module = None 
