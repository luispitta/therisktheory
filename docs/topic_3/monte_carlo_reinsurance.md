# Topic 3: Monte Carlo Simulation, Stability Measures, and Reinsurance Reciprocity

## Mathematical Concepts Implemented

This module implements the core computational layer for Topic 3 within the `therisktheory.severity_aggregate` package:

1. `simulate_aggregate_loss`: High-performance vectorized Monte Carlo simulation for compound sum distributions $S = \sum_{i=1}^N X_i$ without explicit Python loops.
2. `technical_result_variance`: Evaluates the underwriting technical result variance $\text{Var}(R)$ under proportional retention based on compound moment decompositions.
3. `optimal_reinsurance`: Determines the optimal retention proportion $\alpha^* \in [0, 1]$ under variance stability constraints, mean-variance utility, or premium budgets.
4. `insurer_reciprocity_metric`: Analyzes diversification gains and variance reductions achieved through bilateral reciprocity exchanges between direct insurers.

---

## Model Definitions & Mathematical Formulations

### 1. Collective Risk Model & Monte Carlo Simulation

Let the total portfolio aggregate loss over a defined operational period be:

$$S = \sum_{i=1}^N X_i$$

where:
- $N \in \mathbb{N}_0$ is the random number of claims (e.g., $N \sim \text{Poisson}(\lambda)$ or $N \sim \text{NegBin}(r, p)$).
- $X_i > 0$ are independent and identically distributed claim severities (e.g., Lognormal, Pareto, Exponential, Gamma).
- $N$ and $\{X_i\}$ are mutually independent. When $N = 0$, $S = 0$.

#### Vectorized Aggregation Algorithm

For $M$ independent simulation trajectories:
1. Draw claim counts $N_1, \dots, N_M$ from the specified frequency distribution.
2. Compute total claims $N_{\text{total}} = \sum_{j=1}^M N_j$.
3. Draw $N_{\text{total}}$ random claim severities from the specified severity distribution.
4. Construct path indices via `numpy.repeat(np.arange(M), n_claims)`.
5. Accumulate claims into aggregate portfolio totals using `numpy.bincount`.

By the Law of Large Numbers, empirical Monte Carlo moments converge almost surely to the analytical expectations:

$$\mathbb{E}[S] = \mathbb{E}[N]\mathbb{E}[X], \quad \text{Var}(S) = \mathbb{E}[N]\text{Var}(X) + \text{Var}(N)(\mathbb{E}[X])^2$$

---

### 2. Technical Result Variance as a Stability Measure

Under proportional reinsurance with retention rate $\alpha \in [0, 1]$, the insurer retains losses $S_{\text{ret}} = \alpha S$ and pays a deterministic reinsurance premium. The technical underwriting result is $R = P_{\text{ret}} - \alpha S$.

Since $P_{\text{ret}}$ is deterministic, the variance of the technical result is:

$$\text{Var}(R) = \text{Var}(\alpha S) = \alpha^2 \text{Var}(S) = \alpha^2 \left( \mathbb{E}[N]\text{Var}(X) + \text{Var}(N)(\mathbb{E}[X])^2 \right)$$

---

### 3. Optimal Reinsurance Under Stability Constraints

1. **Target Variance Criterion:**
   When the insurer requires $\text{Var}(S_{\text{ret}}) \le V_{\text{target}}$:

   $$\alpha^* = \min\left(1.0, \sqrt{\frac{V_{\text{target}}}{\text{Var}(S)}}\right)$$

2. **Mean-Variance Trade-Off:**
   Minimizing retained variance plus penalized reinsurance cost $(1 - \alpha)(1 + \xi)\mathbb{E}[S]$ with weight $\lambda > 0$:

   $$\alpha^* = \text{clip}\left(\frac{\lambda (1 + \xi)\mathbb{E}[S]}{2 \text{Var}(S)}, 0.0, 1.0\right)$$

3. **Premium Budget Criterion:**
   When constrained by maximum reinsurance expenditure $B$:

   $$\alpha^* = \max\left(0.0, 1.0 - \frac{B}{(1 + \xi)\mathbb{E}[S]}\right)$$

---

### 4. Bilateral Reciprocity Between Insurers

Two direct insurers exchange quotas $q_1, q_2 \in [0, 1]$ of their portfolios $S_1, S_2$. Financial equivalence requires:

$$\mathbb{E}[q_1 S_1] = \mathbb{E}[q_2 S_2] \implies q_2 = q_1 \frac{\mathbb{E}[S_1]}{\mathbb{E}[S_2]}$$

The post-exchange portfolios $Y_1 = (1 - q_1)S_1 + q_2 S_2$ and $Y_2 = q_1 S_1 + (1 - q_2)S_2$ achieve diversification without outward reinsurance expense loading. The system diversification gain is:

$$\mathcal{R}_{\text{system}} = 1 - \frac{\text{Var}(Y_1) + \text{Var}(Y_2)}{\text{Var}(S_1) + \text{Var}(S_2)}$$

---

## Domain & Parameter Bounds

- Simulation count: $M \ge 1$.
- Frequency and severity moments: $\mathbb{E}[N] \ge 0, \text{Var}(N) \ge 0, \mathbb{E}[X] \ge 0, \text{Var}(X) \ge 0$.
- Retention rate: $\alpha \in [0.0, 1.0]$.
- Cession quotas: $q_1, q_2 \in [0.0, 1.0]$.
- Correlation: $\rho \in [-1.0, 1.0]$.

---

## Bibliographic References

- Klugman, S. A., Panjer, H. H., & Willmot, G. E. (2019). *Loss Models: From Data to Decisions* (5th ed.). Wiley, Chapter 9.
- Kaas, R., Goovaerts, M., Dhaene, J., & Denuit, M. (2008). *Modern Actuarial Risk Theory: Using R* (2nd ed.). Springer, Chapters 1–2.
- Bowers, N. L., et al. (1997). *Actuarial Mathematics* (2nd ed.). Society of Actuaries, Chapter 14.
- Beard, R. E., Pentikäinen, T., & Pesonen, E. (1984). *Risk Theory* (3rd ed.). Chapman and Hall, Chapter 6.
