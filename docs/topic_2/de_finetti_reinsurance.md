# Topic 2: De Finetti Theory Applied to Proportional Reinsurance

## Mathematical Concepts Implemented

This module implements Bruno de Finetti's risk and game-theoretic framework applied to proportional reinsurance treaties within the `therisktheory.finetti_ruin` package:

1. `proportional_reinsurance_objective`: Evaluates De Finetti's objective criteria for proportional reinsurance, including the safety index, mean-variance utility, net expected technical profit, and retained claim variance, with mathematically continuous boundary treatment at zero retention.
2. `de_finetti_optimal_retention`: Derives the optimal retention proportion $\alpha^* \in [0, 1]$ under De Finetti's safety index or mean-variance criterion, rigorously contrasting interior solutions with compact domain boundary values.
3. `de_finetti_retention_ruin_approximation` (and alias `de_finetti_retention_ruin_bound`): Evaluates the asymptotic diffusion approximation to ruin probability under proportional reinsurance as a function of initial reserve capital, explicitly distinguished from a proven Cramér-Lundberg analytical bound.

---

## Model Definitions & Mathematical Formulations

### 1. The Proportional Reinsurance Process

Consider an insurance portfolio with gross claim random variable $X$ having mean $\mu = \mathbb{E}[X] > 0$ and variance $\sigma^2 = \text{Var}(X) > 0$. The direct insurer collects gross premium with relative safety loading $\eta \ge 0$:

$$P = (1 + \eta)\mu$$

Under a proportional (quota share) treaty, the direct insurer retains proportion $\alpha \in [0, 1]$ and cedes $1 - \alpha$ to the reinsurer. The reinsurer charges relative safety loading $\xi \ge 0$:

$$P_{\text{re}}(\alpha) = (1 - \alpha)(1 + \xi)\mu$$

The net retained premium of the direct insurer is:

$$P_{\text{ret}}(\alpha) = P - P_{\text{re}}(\alpha) = [(1 + \eta) - (1 - \alpha)(1 + \xi)]\mu = [\alpha(1 + \xi) + (\eta - \xi)]\mu$$

The net expected technical profit (drift) per unit of risk is:

$$\mathbb{E}[\Delta U(\alpha)] = P_{\text{ret}}(\alpha) - \alpha \mu = [\eta - (1 - \alpha)\xi]\mu$$

The variance of retained claims is:

$$\text{Var}(\Delta U(\alpha)) = \text{Var}(\alpha X) = \alpha^2 \sigma^2$$

---

### 2. De Finetti's Safety Index Criterion

De Finetti (1940) formalizes the insurer's objective as maximizing the safety index $\theta(\alpha)$—defined as the ratio of expected net profit (drift) to the variance of results:

$$\theta(\alpha) = \frac{\mathbb{E}[\Delta U(\alpha)]}{\text{Var}(\Delta U(\alpha))} = \frac{[\eta - (1 - \alpha)\xi]\mu}{\alpha^2 \sigma^2} = \frac{\mu}{\sigma^2} \left[ \frac{\eta - \xi}{\alpha^2} + \frac{\xi}{\alpha} \right]$$

---

### 3. Treatment and Mathematical Justification of Zero Retention ($\alpha = 0$)

At zero retention ($\alpha = 0$), the direct insurer cedes 100% of the underwriting portfolio ($1 - \alpha = 1$). The retained variance is identically zero: $\text{Var}(\Delta U(0)) = 0^2 \cdot \sigma^2 = 0$.

Evaluating the objective function at the boundary $\alpha = 0$ requires taking the continuous limit as $\alpha \to 0^+$:

$$\lim_{\alpha \to 0^+} \theta(\alpha) = \lim_{\alpha \to 0^+} \frac{(\eta - \xi)\mu + \alpha \xi \mu}{\alpha^2 \sigma^2}$$

This limit produces distinct mathematical and economic regimes based on the relative loadings $\eta$ and $\xi$:

1. **Reinsurance Arbitrage / Fronting ($\eta > \xi$):**
   - The insurer charges a higher premium loading $\eta$ to policyholders than the reinsurer charges $\xi$.
   - At $\alpha \to 0^+$, the insurer locks in a guaranteed positive net profit drift of $(\eta - \xi)\mu > 0$ with zero retained underwriting risk ($\text{Var} \to 0$).
   - Therefore:
     $$\lim_{\alpha \to 0^+} \theta(\alpha) = \frac{(\eta - \xi)\mu}{0^+} = +\infty$$
   - In extended real numbers $\overline{\mathbb{R}}$, $\theta(0) = +\infty$ (`np.inf`).
   - Consequently, when $\eta > \xi$, full reinsurance ($\alpha^* = 0$) attains the global supremum: $\theta(0) = +\infty \ge \theta(\alpha)$ for all $\alpha \in [0, 1]$.
   - *Example (Instructor Review Benchmark):* For $\mu = 1, \sigma^2 = 1, \eta = 0.20, \xi = 0.10$, at $\alpha = 0.10$ the safety index is $\theta(0.10) = \frac{0.20 - 0.90(0.10)}{0.01} = \frac{0.11}{0.01} = 11.0$. Because $\theta(0) = +\infty > 11.0$, the optimal retention $\alpha^* = 0.0$ strictly maximizes the implemented criterion.

2. **Equal Loadings ($\eta = \xi$):**
   - The expected profit simplifies to $\alpha \eta \mu$.
   - If $\eta > 0$: $\theta(\alpha) = \frac{\alpha \eta \mu}{\alpha^2 \sigma^2} = \frac{\eta \mu}{\alpha \sigma^2} \to +\infty$ as $\alpha \to 0^+$. Thus $\theta(0) = +\infty$ and $\alpha^* = 0.0$.
   - If $\eta = 0$: Expected profit is 0 and variance is 0 for all $\alpha$, so $\theta(0) = 0.0$.

3. **Standard Costly Reinsurance ($\xi > \eta$):**
   - The reinsurer charges more than the primary insurer collects.
   - At $\alpha \to 0^+$, expected profit is negative: $(\eta - \xi)\mu < 0$ (a guaranteed negative cash flow).
   - Therefore:
     $$\lim_{\alpha \to 0^+} \theta(\alpha) = \frac{(\eta - \xi)\mu}{0^+} = -\infty$$
   - In extended real numbers, $\theta(0) = -\infty$ (`-np.inf`). Ceding 100% of risk at a loss produces infinite negative drift per unit of variance.

---

### 4. Boundary and Interior Optimization Analysis

To optimize $\theta(\alpha)$ over the compact interval $\alpha \in [0, 1]$, we differentiate $\theta(\alpha)$ on $(0, 1]$:

$$\frac{d\theta}{d\alpha} = \frac{\mu}{\sigma^2} \left[ -\frac{2(\eta - \xi)}{\alpha^3} - \frac{\xi}{\alpha^2} \right] = \frac{\mu}{\sigma^2 \alpha^3} \left[ 2(\xi - \eta) - \xi \alpha \right]$$

Setting $\frac{d\theta}{d\alpha} = 0$ yields the stationary point:

$$\alpha_0 = 2\left(1 - \frac{\eta}{\xi}\right)$$

The second derivative at $\alpha_0$ is:

$$\theta''(\alpha_0) = -\frac{\mu \xi^4}{8 \sigma^2 (\xi - \eta)^3} < 0 \quad (\text{for } \xi > \eta)$$

which confirms that $\alpha_0$ is a strict local maximum on $(0, \infty)$.

Contrasting the stationary point with the domain boundaries $\alpha \in \{0, 1\}$ gives the global optimum:

| Regime | Condition | Stationary Point $\alpha_0$ | $\theta(0)$ | $\theta(1)$ | Optimum $\alpha^*$ | $\theta(\alpha^*)$ | Optimality Type |
|---|---|---|---|---|---|---|---|
| **I. Arbitrage / Cheap** | $\xi \le \eta$ | $\le 0$ | $+\infty$ | $\frac{\eta \mu}{\sigma^2}$ | $0.0$ | $+\infty$ | Boundary Supremum |
| **II. Moderate Cost** | $\eta < \xi < 2\eta$ | $\in (0, 1)$ | $-\infty$ | $\frac{\eta \mu}{\sigma^2}$ | $2(1 - \frac{\eta}{\xi})$ | $\frac{\mu \xi^2}{4\sigma^2(\xi - \eta)}$ | Interior Strict Global Maximum |
| **III. Expensive** | $\xi \ge 2\eta$ | $\ge 1$ | $-\infty$ | $\frac{\eta \mu}{\sigma^2}$ | $1.0$ | $\frac{\eta \mu}{\sigma^2}$ | Boundary Maximum |

In Regime II, the interior optimum strictly dominates both boundaries because:

$$\theta(\alpha^*) - \theta(1) = \frac{\mu}{\sigma^2} \left[ \frac{\xi^2}{4(\xi - \eta)} - \eta \right] = \frac{\mu (\xi - 2\eta)^2}{4\sigma^2 (\xi - \eta)} > 0$$

In Regime III, $\frac{d\theta}{d\alpha} > 0$ for all $\alpha \in (0, 1)$, so $\theta(\alpha)$ is strictly increasing on $(0, 1]$, and $\alpha^* = 1.0$ is the unique maximum on $[0, 1]$.

---

### 5. Mean-Variance Criterion

Under the mean-variance utility framework with risk aversion parameter $k > 0$:

$$\Phi(\alpha) = \mathbb{E}[\Delta U(\alpha)] - \frac{k}{2} \text{Var}(\Delta U(\alpha)) = [\eta - (1 - \alpha)\xi]\mu - \frac{k}{2} \alpha^2 \sigma^2$$

Setting the marginal condition $\frac{d\Phi}{d\alpha} = \xi \mu - k \alpha \sigma^2 = 0$:

$$\alpha^* = \text{clip}\left(\frac{\xi \mu}{k \sigma^2}, 0.0, 1.0\right)$$

---

### 6. Diffusion Approximation vs. Proven Cramér-Lundberg Bound

In classical ruin theory, it is essential to distinguish between a **proven analytical upper bound** and a **diffusion (heavy-traffic) approximation**:

#### A. Cramér-Lundberg Upper Bound (Proven Analytical Inequality)
In the classical compound Poisson risk model with premium rate $c = (1 + \eta)\mu$, the Lundberg inequality states:

$$\psi(u) \le \exp(-R u) \quad \forall u \ge 0$$

where $R > 0$ is the unique positive root of the exact Cramér-Lundberg equation:

$$\mathbb{E}[e^{R X}] = 1 + (1 + \eta) \mu R$$

This is a **mathematically proven upper bound** established via martingale inequalities and renewal theory. It holds for all claim distributions with finite moment-generating functions in a neighborhood of zero.

#### B. De Finetti's Diffusion Approximation (Brownian Motion Proxy)
Bruno de Finetti (1940) approximated the discrete or compound Poisson surplus process by a continuous Brownian motion with matching first two moments:
- Drift $m(\alpha) = \mathbb{E}[\Delta U(\alpha)]$
- Variance $s^2(\alpha) = \text{Var}(\Delta U(\alpha))$

For a Brownian motion with drift $m > 0$ and diffusion coefficient $s^2$, the first-passage ruin probability from level $u \ge 0$ is given exactly by:

$$\psi_{\text{diff}}(u; \alpha) = \exp\left( - \frac{2 m(\alpha)}{s^2(\alpha)} u \right) = \exp\left( - 2 \theta(\alpha) u \right)$$

where $2\theta(\alpha) = R_{\text{diff}}$ is the diffusion adjustment coefficient (the second-order Taylor expansion of $R$).

#### Key Distinctions
1. **Not an Upper Bound:** For heavy-tailed or non-exponential claims, $\psi_{\text{diff}}(u) = \exp(-2\theta u)$ may either **underestimate or overestimate** the exact ruin probability $\psi(u)$. It is an asymptotic proxy (valid under heavy traffic as $\eta \to 0$ and $u \to \infty$), not a demonstrated upper bound.
2. **Implementation Naming:**
   - The primary function is named `de_finetti_retention_ruin_approximation`.
   - A backward-compatible alias `de_finetti_retention_ruin_bound` is provided with clear documentation distinguishing the diffusion proxy from a Lundberg inequality.

---

## Domain & Parameter Restrictions

- Retention proportion: $\alpha \in [0.0, 1.0]$.
- Claim distribution moments: $\mu > 0, \sigma^2 > 0$.
- Relative loadings: $\eta \ge 0, \xi \ge 0$.
- Risk aversion parameter: $k > 0$.
- Initial capital reserve: $u \ge 0$.

---

## Bibliographic References

- **De Finetti, B. (1940).** *Il rischio nel giuoco e nelle assicurazioni*. IV Congresso dell'Unione Matematica Italiana, Sections 1–3. (Foundational formulation of the safety index and proportional retention game).
- **Beard, R. E., Pentikäinen, T., & Pesonen, E. (1984).** *Risk Theory* (3rd ed.). Chapman and Hall, Chapter 6, Section 6.4. (Analytical derivation of De Finetti's optimal retention and boundary regimes).
- **Bühlmann, H. (1970).** *Mathematical Methods in Risk Theory*. Springer-Verlag, Chapter 5. (Risk exchange, quadratic utility, and ruin criteria).
- **Asmussen, S., & Albrecher, H. (2010).** *Ruin Probabilities* (2nd ed.). World Scientific, Chapter 4. (Diffusion approximations vs. exact Lundberg bounds and heavy-traffic limits).
