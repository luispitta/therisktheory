"""De Finetti games, safety and ruin models."""

from importlib import import_module

_MODULES = ['games','adjustment','round_loading','de_finetti_safety','de_finetti_ruin','de_finetti_reinsurance','de_finetti_comparison']


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
