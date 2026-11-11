# Topic 3: Estimation of Aggregate Loss Distribution

Package: `therisktheory.collective_risk.approximations`. Status: experimental.
References refer to Kaas, Goovaerts, Dhaene and Denuit (2008), *Modern Actuarial Risk Theory: Using R*, 2nd ed., Springer; and Daykin, Pentikäinen and Pesonen (1994), *Practical Risk Theory for Actuaries*, Chapman & Hall.

## Mathematical Formulation
Let $S = \sum_{i=1}^N X_i$ be the aggregate claim amount with mean $\mu$, variance $\sigma^2$, and skewness $\gamma_1 = \mathbb{E}[(S-\mu)^3] / \sigma^3$.
Let $y = (x - \mu) / \sigma$.

### 1. Normal Approximation
$$F_S(x) \approx \Phi(y)$$

### 2. Edgeworth Expansion (First Order)
$$F_S(x) \approx \Phi(y) - \frac{\gamma_1}{6}(y^2 - 1)\phi(y)$$
(Kaas et al. 2008, eq. 3.54, p. 58).

### 3. Normal Power (NP) Approximation
Solves the quadratic relation $y = z + \frac{\gamma_1}{6}(z^2 - 1)$:
$$z = \sqrt{\frac{9}{\gamma_1^2} + 1 + \frac{6y}{\gamma_1}} - \frac{3}{\gamma_1}, \quad F_S(x) \approx \Phi(z)$$
(Kaas et al. 2008, eq. 3.58, p. 59).
