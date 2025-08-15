"""Our contract implementation built on top of BaseContract."""
from .base import BaseContract


class OurContract(BaseContract):
    """Default contract used in simulations."""
    pass

__all__ = ["OurContract"]
