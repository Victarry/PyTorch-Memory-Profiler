import torch
import importlib
import logging

from .base_plugin import TracerPlugin

# Configure logging
logger = logging.getLogger(__name__)

# List of modules to patch for distributed functions
target_modules = [
    "megatron.core.tensor_parallel.mappings",
    "megatron.core.distributed.param_and_grad_buffer",
    "megatron.core.tensor_parallel.layers",
    "megatron.core.timers",
]


class MegatronCorePlugin(TracerPlugin):
    """
    Plugin that patches Megatron-LM core modules to force specific implementations
    of distributed communication, multi-tensor operations, and optimizer functions.
    """

    def setup(self, tracer):
        super().setup(tracer)
        logger.info("Setting up MegatronCorePlugin")

    def enter(self):
        """Apply patches when entering the context manager"""
        logger.info("Applying Megatron-LM core patches")
        apply_megatron_core_patch()

    def exit(self, exc_type, exc_val, exc_tb):
        """Method called when exiting the context manager"""
        logger.info("MegatronCorePlugin context exited")


def apply_megatron_core_patch():
    """
    Patches Megatron-LM core modules to forcibly use specific
    distributed communication functions and implementations.
    This includes:
    1. Distributed communication functions (_all_gather_base, _reduce_scatter_base)
    2. Multi-tensor implementations (forcing local implementation)
    3. Adam optimizer (replacing with AdamW)
    4. Replacing _coalescing_manager with nullcontext to fix related errors
    """
    # Apply all patches
    patch_distributed_functions()
    patch_multi_tensor_functions()
    patch_adam_optimizer()
    patch_coalescing_manager()

    logger.info("All Megatron-LM core patches have been applied.")


def patch_distributed_functions():
    """
    Patches Megatron-LM core modules to forcibly use specific
    distributed communication functions (_all_gather_base, _reduce_scatter_base),
    regardless of the torch version check results in the original code.
    """
    if not hasattr(torch.distributed, "_all_gather_base") or not hasattr(
        torch.distributed, "_reduce_scatter_base"
    ):
        logger.warning(
            "torch.distributed._all_gather_base or "
            "torch.distributed._reduce_scatter_base not found. "
            "Skipping Megatron-LM core distributed function patch."
        )
        return

    all_gather_func = torch.distributed._all_gather_base
    reduce_scatter_func = torch.distributed._reduce_scatter_base

    patched_something = False
    for module_name in target_modules:
        try:
            module = importlib.import_module(module_name)
            module_patched = False

            if hasattr(module, "dist_all_gather_func"):
                if getattr(module, "dist_all_gather_func") != all_gather_func:
                    setattr(module, "dist_all_gather_func", all_gather_func)
                    logger.info(f"Patched dist_all_gather_func in {module_name}")
                    module_patched = True
                else:
                    logger.debug(f"dist_all_gather_func already set in {module_name}")
            else:
                logger.warning(
                    f"Attribute dist_all_gather_func not found in {module_name}"
                )

            if hasattr(module, "dist_reduce_scatter_func"):
                if getattr(module, "dist_reduce_scatter_func") != reduce_scatter_func:
                    setattr(module, "dist_reduce_scatter_func", reduce_scatter_func)
                    logger.info(f"Patched dist_reduce_scatter_func in {module_name}")
                    module_patched = True
                else:
                    logger.debug(
                        f"dist_reduce_scatter_func already set in {module_name}"
                    )
            else:
                logger.warning(
                    f"Attribute dist_reduce_scatter_func not found in {module_name}"
                )

            if module_patched:
                patched_something = True

        except ImportError:
            logger.warning(
                f"Module {module_name} not found. Skipping patch for this module."
            )
        except Exception as e:
            logger.error(f"Error patching module {module_name}: {e}", exc_info=True)

    if patched_something:
        logger.info(
            "Successfully applied Megatron-LM core patches for distributed functions."
        )
    else:
        logger.info(
            "Megatron-LM core patching for distributed functions did not modify any modules (either already patched or attributes not found)."
        )


def patch_multi_tensor_functions():
    """
    Patches Megatron-LM core optimizer/clip_grads.py to forcibly use
    local multi-tensor implementations, bypassing Transformer Engine and APEX.
    """
    try:
        # Import the clip_grads module
        clip_grads_module = importlib.import_module(
            "megatron.core.optimizer.clip_grads"
        )

        # Import local implementations
        utils_module = importlib.import_module("megatron.core.utils")

        # Check if local implementations exist
        if not all(
            hasattr(utils_module, attr)
            for attr in [
                "local_multi_tensor_applier",
                "local_multi_tensor_l2_norm",
                "local_multi_tensor_scale",
            ]
        ):
            logger.warning(
                "Local multi-tensor implementations not found in megatron.core.utils"
            )
            return

        # Get local implementations
        local_applier = utils_module.local_multi_tensor_applier
        local_l2_norm = utils_module.local_multi_tensor_l2_norm
        local_scale = utils_module.local_multi_tensor_scale

        # Force set the implementations
        setattr(clip_grads_module, "multi_tensor_applier", local_applier)
        setattr(clip_grads_module, "l2_norm_impl", local_l2_norm)
        setattr(clip_grads_module, "multi_tensor_scale_impl", local_scale)

        logger.info(
            "Successfully patched multi_tensor functions in megatron.core.optimizer.clip_grads"
        )

    except ImportError:
        logger.warning(
            "Module megatron.core.optimizer.clip_grads not found. Skipping multi-tensor patch."
        )
    except Exception as e:
        logger.error(f"Error patching multi-tensor functions: {e}", exc_info=True)


def patch_adam_optimizer():
    """
    Patches Megatron-LM's distributed optimizer to use torch.optim.AdamW
    instead of the default Adam implementation.
    """
    try:
        # Import the distrib_optimizer module
        distrib_optimizer_module = importlib.import_module(
            "megatron.core.optimizer.distrib_optimizer"
        )

        # Force set AdamW
        if hasattr(torch.optim, "AdamW"):
            setattr(distrib_optimizer_module, "Adam", torch.optim.AdamW)
            logger.info(
                "Successfully patched Adam to use torch.optim.AdamW in megatron.core.optimizer.distrib_optimizer"
            )
        else:
            logger.warning("torch.optim.AdamW not found. Skipping Adam patch.")

    except ImportError:
        logger.warning(
            "Module megatron.core.optimizer.distrib_optimizer not found. Skipping Adam patch."
        )
    except Exception as e:
        logger.error(f"Error patching Adam optimizer: {e}", exc_info=True)


def patch_coalescing_manager():
    """
    Patches PyTorch's _coalescing_manager to return a DummyWorkHandle in Megatron-LM modules,
    """
    try:
        # Create a dummy work handle class with wait method
        class DummyWorkHandle:
            def __init__(self):
                pass

            def wait(self):
                """Dummy wait method that does nothing but returns successfully"""
                logger.debug("Dummy work handle wait() called")
                return True

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

            def append(self, work):
                """Handle append method that _coalescing_manager uses"""
                logger.debug("Dummy work handle append() called")
                return

        # Create a simple replacement function that returns a context with wait method
        def safe_coalescing_manager(*args, **kwargs):
            logger.debug(
                "Using safe replacement for _coalescing_manager with wait() support"
            )
            # Return a dummy handle that will be used as context manager and later for wait()
            return DummyWorkHandle()

        # Patch the param_and_grad_buffer module
        try:
            param_and_grad_buffer = importlib.import_module(
                "megatron.core.distributed.param_and_grad_buffer"
            )
            if hasattr(param_and_grad_buffer, "_coalescing_manager"):
                setattr(
                    param_and_grad_buffer,
                    "_coalescing_manager",
                    safe_coalescing_manager,
                )
                logger.info(
                    "Successfully patched _coalescing_manager in param_and_grad_buffer"
                )
            else:
                # If _coalescing_manager is imported directly, we need to patch the import
                import torch.distributed

                original_coalescing_manager = torch.distributed._coalescing_manager

                # Only patch it when called from Megatron
                def patched_coalescing_manager(*args, **kwargs):
                    # Check the call stack to determine if this is called from megatron code
                    import traceback

                    call_stack = traceback.extract_stack()
                    is_from_megatron = any(
                        "megatron" in frame.filename for frame in call_stack
                    )

                    if is_from_megatron:
                        logger.debug(
                            "Using dummy work handle for _coalescing_manager called from Megatron"
                        )
                        return DummyWorkHandle()
                    else:
                        return original_coalescing_manager(*args, **kwargs)

                torch.distributed._coalescing_manager = patched_coalescing_manager
                logger.info(
                    "Successfully patched torch.distributed._coalescing_manager for Megatron calls"
                )
        except ImportError:
            logger.warning(
                "Module megatron.core.distributed.param_and_grad_buffer not found"
            )
        except Exception as e:
            logger.error(
                f"Error patching _coalescing_manager in param_and_grad_buffer: {e}",
                exc_info=True,
            )

    except Exception as e:
        logger.error(f"Error in patch_coalescing_manager: {e}", exc_info=True)


# To use this plugin, import it and call apply_megatron_core_patch
# during the initialization of your memory profiling setup, or register
# the MegatronCorePlugin with your MemoryTracer instance:
# tracer.register_plugin(MegatronCorePlugin())
