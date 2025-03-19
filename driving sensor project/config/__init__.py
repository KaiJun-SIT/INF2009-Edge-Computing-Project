# config/__init__.py
from .config_loader import ConfigLoader

# Create a singleton instance
config = ConfigLoader()

__all__ = ['config']