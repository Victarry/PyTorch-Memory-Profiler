from abc import ABC, abstractmethod

class TracerPlugin(ABC):
    """Base abstract class for MemoryTracer plugins."""
    
    @abstractmethod
    def setup(self, tracer):
        """Setup the plugin with the tracer instance."""
        pass
    
    @abstractmethod
    def enter(self):
        """Called when entering the tracer context."""
        pass
    
    @abstractmethod
    def exit(self, exc_type, exc_val, exc_tb):
        """Called when exiting the tracer context."""
        pass 