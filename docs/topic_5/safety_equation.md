# Topic 5: Safety Equation and Lundberg Fundamental Equation

## 1. Theoretical Scope and Motivation
Collective risk theory models the continuous progression of an insurer's surplus over time. A central element in evaluating ruin probability is the **adjustment coefficient** $R$ (Lundberg's exponent), which measures the exponential decay rate of the ruin probability $\psi(u) \le e^{-R u}$. This module integrates frequency structures (Bracamonte) and severity moment calculations (Garcia) into a unified collective risk framework.

## 2. Mathematical Formulation
Let $X$ represent the random claim severity with moment generating function $M_X(r) = E[e^{r X}]$ and mean $E[X]$. Let $\lambda > 0$ denote the insurer's relative safety loading.

* **Lundberg's Fundamental Equation:**
  The adjustment coefficient $R > 0$ is defined as the unique positive solution to:
  $$M_X(R) - 1 - (1 + \lambda) E[X] R = 0$$

* **Full Retention Approximation (Moments Integration):**
  Using Garcia's severity moment implementations for $E[X]$ and $E[X^2]$, the adjustment coefficient under full retention is approximated as:
  $$R \approx \frac{2 \lambda E[X]}{E[X^2]}$$

## 3. Bibliographic References
* Beard, R.; Pentikäinen, T.; Pesonen, E. — *Risk Theory*[cite: 1].
* Bühlmann, H. — *Mathematical Methods in Risk Theory*[cite: 1].
* Gerber, H. — *An Introduction to Mathematical Risk Theory*[cite: 1].
