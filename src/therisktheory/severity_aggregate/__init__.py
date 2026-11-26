"""Claim severity, transformations and simulation models."""

from importlib import import_module

_MODULES = ['lognormal','pareto','weibull','beta','transformations','monte_carlo','reinsurance_stability']


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


__all__ = ['lognormal_severity_pdf', 'lognormal_severity_cdf', 'lognormal_severity_ppf', 'lognormal_severity_moments', 'pareto_severity_pdf', 'pareto_severity_cdf', 'pareto_severity_ppf', 'pareto_severity_moments', 'weibull_severity_pdf', 'weibull_severity_cdf', 'weibull_severity_ppf', 'weibull_severity_moments', 'beta_severity_pdf', 'beta_severity_cdf', 'beta_severity_ppf', 'beta_severity_moments']
