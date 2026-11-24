"""Collective risk process, approximations, Lundberg and insolvency models."""

from importlib import import_module

_MODULES = ['aggregate_loss','safety_equation','process','approximations','reinsurance_effects','stability','insolvency']


def __getattr__(name: str):
    """Resolve contributed APIs without eagerly loading unrelated modules."""
    if name.startswith("_"):
        raise AttributeError(name)
    if name in _MODULES:
        result = import_module(f".{name}", __name__)
        globals()[name] = result
        return result
    for module_name in _MODULES:
        try:
            module = import_module(f".{module_name}", __name__)
            result = getattr(module, name)
        except (ImportError, AttributeError):
            continue
        globals()[name] = result
        return result
    raise AttributeError(f"{__name__} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_MODULES))
