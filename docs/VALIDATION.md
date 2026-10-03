# Mathematical Validation Standard

The repository distinguishes software correctness from mathematical validity.

## Validation states

### Experimental
The implementation exists and automated tests may pass, but the mathematics has not completed peer review.

### Reviewed
A second contributor has checked the implementation, assumptions, derivation, and tests.

### Mathematically validated
The maintainer/instructor has confirmed:

1. the mathematical statement is correct;
2. assumptions and domain restrictions are explicit;
3. the bibliographic source is appropriate;
4. the implementation matches the formula;
5. analytical tests are correct;
6. numerical validation is adequate;
7. edge cases are documented.

AI-generated explanations, derivations, or tests are not acceptable as primary mathematical evidence.

## Minimum evidence for a mathematical function

Every function should document:

- notation;
- mathematical formula;
- assumptions;
- valid parameter domain;
- units when relevant;
- reference;
- expected numerical behavior;
- known limitations.

## Numerical comparison

Use explicit tolerances.

```python
np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
```

The tolerance must be justified when materially looser than standard floating-point expectations.
