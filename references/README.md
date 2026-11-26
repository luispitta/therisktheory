# References

Record bibliographic sources used to validate mathematical implementations here.

Do not upload copyrighted books or papers unless redistribution is permitted.

Prefer bibliographic metadata and stable links/DOIs where available.

## Sources

### Kaas, Goovaerts, Dhaene and Denuit (2008)

Kaas, R., Goovaerts, M., Dhaene, J. and Denuit, M. (2008). *Modern Actuarial Risk Theory:
Using R*, 2nd ed. Springer. ISBN 978-3-540-70992-3, e-ISBN 978-3-540-70998-5.
DOI: [10.1007/978-3-540-70998-5](https://doi.org/10.1007/978-3-540-70998-5)
(metadata checked against Crossref).

| Used for | Location |
|---|---|
| Collective model and independence assumptions | Section 3.1, eq. (3.1), p. 41 |
| `expected_aggregate_loss`: `E[S] = E[N] E[X]` | Section 3.2, eqs. (3.2)–(3.3), pp. 42–43 |
| `pure_premium`: net premium / equivalence principle | Section 5.3, principle (a), p. 119 |
| `expected_value_premium`: `(1 + θ) E[S]` | Section 5.3, principle (b), p. 119 |
| Properties of the expected value principle | Section 5.3.1, pp. 120–121; Table 5.1, p. 122 |

Module: `therisktheory.solvency_pricing.premiums` (Topic 1).

### Bühlmann, H. (1970)
Bühlmann, H. (1970). *Mathematical Methods in Risk Theory*. Springer.

| Used for | Location |
|---|---|
| Credibility theory and Bayesian risk premium foundations | Chapter 4 |

### Klugman, S. A., Panjer, H. H., & Willmot, G. E. (2019)
Klugman, S. A., Panjer, H. H., & Willmot, G. E. (2019). *Loss Models: From Data to Decisions* (5th ed.). Wiley.

| Used for | Location |
|---|---|
| Gamma-Poisson conjugacy, posterior update, and predictive mean | Chapter 16 |

### Bowers, N. L., Gerber, H. U., Hickman, J. C., Jones, D. A., & Nesbitt, C. J. (1997)
Bowers, N. L., Gerber, H. U., Hickman, J. C., Jones, D. A., & Nesbitt, C. J. (1997). *Actuarial Mathematics* (2nd ed.). Society of Actuaries.

| Used for | Location |
|---|---|
| Frequency modeling and exposure-based claim rates | Chapters 11–12 |
