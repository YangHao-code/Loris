"""Calibrated neural model pools and staged original-TPE rule discovery."""

# Import these lazily: regular Loris use does not require the optional BO/Torch extras.
__all__ = ['discover', 'fit_pool', 'extend_context']


def __getattr__(name):
    if name == 'discover':
        from .pipeline import discover
        return discover
    if name in ('fit_pool', 'extend_context'):
        from . import pool
        return getattr(pool, name)
    raise AttributeError(name)
