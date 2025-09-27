try:
    # Prefer your custom implementation by default
    from .msdd.msdd_custom import MSDDDiarizer
except Exception:
    # Fallback to the builtin if custom isn’t present for some reason
    from .msdd.msdd import MSDDDiarizer

__all__ = ["MSDDDiarizer"]
