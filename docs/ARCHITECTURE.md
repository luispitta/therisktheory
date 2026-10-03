# Architecture

## Canonical mathematical layer

NumPy and SciPy are the canonical numerical implementation layer.

The core must not import pandas, Polars, or PySpark.

This keeps mathematical functions:

- reusable;
- vectorized;
- easy to test;
- lightweight;
- independent from tabular execution engines.

## Backend compatibility

### pandas

Convert Series/columns to NumPy arrays when this is efficient and return results while preserving index semantics in an adapter.

### Polars

Prefer native Polars expressions for simple algebra. For functions that require the NumPy reference implementation, execute in batches rather than row-by-row Python callbacks.

### Spark

Prefer Spark SQL/native column expressions.

For numerical routines that cannot be expressed natively, use Arrow-based vectorized execution. Never collect production-scale Spark data to the driver to call a NumPy function.

## Rule

The NumPy implementation defines the numerical behavior.

Backend-specific implementations must be tested for parity against it.
