import torch
import functools
from .base_plugin import TracerPlugin


class P2PCommunicationPlugin(TracerPlugin):
    """
    Plugin for monitoring memory usage in P2P communication operations for pipeline parallelism.
    This plugin tracks memory usage in functions like send_forward_recv_backward and other P2P operations
    used in Megatron-LM's pipeline parallel implementation.
    """

    def setup(self, tracer):
        """Setup the plugin with the tracer instance."""
        self.tracer = tracer
        self.original_funcs = {}

    def enter(self):
        """Patch P2P communication functions when entering the tracer context."""
        try:
            # Dynamically import Megatron-LM's P2P communication module
            import megatron.core.pipeline_parallel.p2p_communication as p2p_comm

            # Import core for pipeline_state check
            import megatron.core as core

            # Patch send_forward_backward_recv_forward_backward
            if hasattr(p2p_comm, "send_forward_backward_recv_forward_backward"):
                original_func = p2p_comm.send_forward_backward_recv_forward_backward
                self.original_funcs["send_forward_backward_recv_forward_backward"] = (
                    original_func
                )
                p2p_comm.send_forward_backward_recv_forward_backward = (
                    self._patch_send_forward_backward_recv_forward_backward(
                        original_func
                    )
                )

            # Patch send_forward_recv_backward
            if hasattr(p2p_comm, "send_forward_recv_backward"):
                original_func = p2p_comm.send_forward_recv_backward
                self.original_funcs["send_forward_recv_backward"] = original_func
                p2p_comm.send_forward_recv_backward = (
                    self._patch_send_forward_recv_backward(original_func)
                )

            # Patch send_backward_recv_forward
            if hasattr(p2p_comm, "send_backward_recv_forward"):
                original_func = p2p_comm.send_backward_recv_forward
                self.original_funcs["send_backward_recv_forward"] = original_func
                p2p_comm.send_backward_recv_forward = (
                    self._patch_send_backward_recv_forward(original_func)
                )

            # Patch send_forward_recv_forward
            if hasattr(p2p_comm, "send_forward_recv_forward"):
                original_func = p2p_comm.send_forward_recv_forward
                self.original_funcs["send_forward_recv_forward"] = original_func
                p2p_comm.send_forward_recv_forward = (
                    self._patch_send_forward_recv_forward(original_func)
                )

            # Patch send_backward_recv_backward
            if hasattr(p2p_comm, "send_backward_recv_backward"):
                original_func = p2p_comm.send_backward_recv_backward
                self.original_funcs["send_backward_recv_backward"] = original_func
                p2p_comm.send_backward_recv_backward = (
                    self._patch_send_backward_recv_backward(original_func)
                )

            # Patch recv_forward
            if hasattr(p2p_comm, "recv_forward"):
                original_func = p2p_comm.recv_forward
                self.original_funcs["recv_forward"] = original_func
                p2p_comm.recv_forward = self._patch_recv_forward(original_func)

            # Patch recv_backward
            if hasattr(p2p_comm, "recv_backward"):
                original_func = p2p_comm.recv_backward
                self.original_funcs["recv_backward"] = original_func
                p2p_comm.recv_backward = self._patch_recv_backward(original_func)

            # Patch send_forward
            if hasattr(p2p_comm, "send_forward"):
                original_func = p2p_comm.send_forward
                self.original_funcs["send_forward"] = original_func
                p2p_comm.send_forward = self._patch_send_forward(original_func)

            # Patch send_backward
            if hasattr(p2p_comm, "send_backward"):
                original_func = p2p_comm.send_backward
                self.original_funcs["send_backward"] = original_func
                p2p_comm.send_backward = self._patch_send_backward(original_func)

        except ImportError:
            # Megatron may not be installed or accessible
            pass

    def _has_fake_tensor(self, *args, **kwargs):
        """Helper method to check if any tensors are fake tensors."""
        from torch._subclasses.fake_tensor import FakeTensor

        # Check args for fake tensors
        has_fake_tensor = any(
            isinstance(arg, FakeTensor) for arg in args if isinstance(arg, torch.Tensor)
        )

        # Check kwargs for fake tensors
        has_fake_tensor = has_fake_tensor or any(
            isinstance(arg, FakeTensor)
            for arg in kwargs.values()
            if isinstance(arg, torch.Tensor)
        )

        return has_fake_tensor

    def _patch_send_forward_backward_recv_forward_backward(self, orig_func):
        """Create patched version of send_forward_backward_recv_forward_backward function."""

        @functools.wraps(orig_func)
        def patched_func(
            output_tensor, input_tensor_grad, recv_prev, recv_next, tensor_shape, config
        ):
            # TODO: check the tensor shape implementation
            # Handle the dual output case - always use fake tensors
            if tensor_shape is not None:
                config_dtype = (
                    getattr(config, "pipeline_dtype", torch.float32)
                    if config
                    else torch.float32
                )
                return (
                    (
                        self.tracer.create_fake_tensor(
                            *tensor_shape, dtype=config_dtype, requires_grad=True
                        )
                        if recv_prev
                        else None
                    ),
                    (
                        self.tracer.create_fake_tensor(
                            *tensor_shape, dtype=config_dtype, requires_grad=True
                        )
                        if recv_next
                        else None
                    ),
                )
            return None, None

        return patched_func

    def _patch_send_forward_recv_backward(self, orig_func):
        """Create patched version of send_forward_recv_backward function."""

        @functools.wraps(orig_func)
        def patched_func(output_tensor, tensor_shape, config):
            # Import needed module inside the function
            import megatron.core as core

            if core.parallel_state.is_pipeline_last_stage():
                return None
            else:
                config_dtype = (
                    getattr(config, "pipeline_dtype", torch.float32)
                    if config
                    else torch.float32
                )
                return self.tracer.create_fake_tensor(
                    *tensor_shape, dtype=config_dtype, requires_grad=True
                )

        return patched_func

    def _patch_send_backward_recv_forward(self, orig_func):
        """Create patched version of send_backward_recv_forward function."""

        @functools.wraps(orig_func)
        def patched_func(input_tensor_grad, tensor_shape, config):
            # Import needed module inside the function
            import megatron.core as core

            # Always use fake tensors
            if tensor_shape is not None:
                config_dtype = (
                    getattr(config, "pipeline_dtype", torch.float32)
                    if config
                    else torch.float32
                )
                # Only create input tensor if not in pipeline first stage
                if not core.parallel_state.is_pipeline_first_stage():
                    return self.tracer.create_fake_tensor(
                        *tensor_shape, dtype=config_dtype, requires_grad=True
                    )
            return None

        return patched_func

    def _patch_send_forward_recv_forward(self, orig_func):
        """Create patched version of send_forward_recv_forward function."""

        @functools.wraps(orig_func)
        def patched_func(
            output_tensor, recv_prev, tensor_shape, config, overlap_p2p_comm=False
        ):
            # Always use fake tensors
            if tensor_shape is not None:
                config_dtype = (
                    getattr(config, "pipeline_dtype", torch.float32)
                    if config
                    else torch.float32
                )
                fake_tensor = (
                    self.tracer.create_fake_tensor(*tensor_shape, dtype=config_dtype, requires_grad=True)
                    if recv_prev
                    else None
                )

                if overlap_p2p_comm:
                    # Create fake wait handles for overlapping communication
                    class FakeWaitHandle:
                        def wait(self):
                            pass

                    fake_handles = {}
                    if output_tensor is not None:  # If we're sending forward
                        fake_handles["send_next"] = FakeWaitHandle()
                    if recv_prev:
                        fake_handles["recv_prev"] = FakeWaitHandle()

                    return fake_tensor, fake_handles
                return fake_tensor
            return None

        return patched_func

    def _patch_send_backward_recv_backward(self, orig_func):
        """Create patched version of send_backward_recv_backward function."""

        @functools.wraps(orig_func)
        def patched_func(
            input_tensor_grad, recv_next, tensor_shape, config, overlap_p2p_comm=False
        ):
            # Always use fake tensors
            if tensor_shape is not None:
                config_dtype = (
                    getattr(config, "pipeline_dtype", torch.float32)
                    if config
                    else torch.float32
                )
                fake_tensor = (
                    self.tracer.create_fake_tensor(*tensor_shape, dtype=config_dtype, requires_grad=True)
                    if recv_next
                    else None
                )

                if overlap_p2p_comm:
                    # Create fake wait handles for overlapping communication
                    class FakeWaitHandle:
                        def wait(self):
                            pass

                    fake_handles = {}
                    if input_tensor_grad is not None:  # If we're sending backward
                        fake_handles["send_prev"] = FakeWaitHandle()
                    if recv_next:
                        fake_handles["recv_next"] = FakeWaitHandle()

                    return fake_tensor, fake_handles
                return fake_tensor
            return None

        return patched_func

    def _patch_recv_forward(self, orig_func):
        """Create patched version of recv_forward function."""

        @functools.wraps(orig_func)
        def patched_func(tensor_shape, config):
            # Import needed module inside the function
            import megatron.core as core

            # Skip original function call if first pipeline stage
            if core.parallel_state.is_pipeline_first_stage():
                return None

            # Always create a fake tensor
            if tensor_shape is not None:
                config_dtype = (
                    getattr(config, "pipeline_dtype", torch.float32)
                    if config
                    else torch.float32
                )
                # Create a fake tensor for the input
                return self.tracer.create_fake_tensor(*tensor_shape, dtype=config_dtype, requires_grad=True)

            return None

        return patched_func

    def _patch_recv_backward(self, orig_func):
        """Create patched version of recv_backward function."""

        @functools.wraps(orig_func)
        def patched_func(tensor_shape, config):
            # Import needed module inside the function
            import megatron.core as core

            # Skip original function call if last pipeline stage
            if core.parallel_state.is_pipeline_last_stage():
                return None

            # Always create a fake tensor
            if tensor_shape is not None:
                config_dtype = (
                    getattr(config, "pipeline_dtype", torch.float32)
                    if config
                    else torch.float32
                )
                # Create a fake tensor for the output grad
                return self.tracer.create_fake_tensor(*tensor_shape, dtype=config_dtype, requires_grad=True)

            return None

        return patched_func

    def _patch_send_forward(self, orig_func):
        """Create patched version of send_forward function."""

        @functools.wraps(orig_func)
        def patched_func(output_tensor, config):
            # Import needed module inside the function
            import megatron.core as core

            # Skip if last pipeline stage
            if core.parallel_state.is_pipeline_last_stage():
                return None

            # Always return None for send operations
            return None

        return patched_func

    def _patch_send_backward(self, orig_func):
        """Create patched version of send_backward function."""

        @functools.wraps(orig_func)
        def patched_func(input_tensor_grad, config):
            # Import needed module inside the function
            import megatron.core as core

            # Skip if first pipeline stage
            if core.parallel_state.is_pipeline_first_stage():
                return None

            # Always return None for send operations
            return None

        return patched_func

    def exit(self, exc_type, exc_val, exc_tb):
        """Restore original functions when exiting the tracer context."""
        try:
            import megatron.core.pipeline_parallel.p2p_communication as p2p_comm

            # Restore original functions
            for func_name, orig_func in self.original_funcs.items():
                setattr(p2p_comm, func_name, orig_func)

        except ImportError:
            pass
