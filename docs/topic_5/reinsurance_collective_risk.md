# Reinsurance Effects on the Collective Risk Process and Ruin Theory

## Mathematical Foundations & Model Formulation

In classical collective risk theory, the insurer's surplus process $\{U(t)\}_{t \ge 0}$ is modeled as a continuous-time compound Poisson process:

$$U(t) = u + c t - \sum_{i=1}^{N(t)} X_i$$

where:
- $u \ge 0$ is the initial capital reserve.
- $\{N(t)\}_{t \ge 0}$ is a homogeneous Poisson claim arrival process with intensity $\lambda > 0$.
- $\{X_i\}_{i \ge 1}$ are independent and identically distributed (i.i.d.) positive random variables representing individual claim severities with common distribution $F_X(x)$, mean $\mathbb{E}[X] = \mu_1$, variance $\mathrm{Var}(X) = \sigma^2$, and moment-generating function $m_X(r) = \mathbb{E}[e^{rX}]$.
- $c = (1 + \theta) \lambda \mu_1$ is the gross premium rate received per unit time, where $\theta > 0$ denotes the insurer's relative safety loading.

When an insurer enters into a reinsurance treaty, a portion of every claim $X_i$ is ceded to a reinsurer. The total claim is partitioned into retained and ceded components:

$$X_i = X_{i, \text{ret}} + X_{i, \text{ced}}$$

The reinsurer demands a premium rate calculated under the expected value principle with safety loading $\xi \ge 0$:

$$c_{\text{re}} = (1 + \xi) \lambda \mathbb{E}[X_{\text{ced}}]$$

The insurer's net retained premium rate is therefore:

$$c_{\text{ret}} = c - c_{\text{re}} = (1 + \theta) \lambda \mu_1 - (1 + \xi) \lambda \mathbb{E}[X_{\text{ced}}]$$

and the net retained surplus process becomes:

$$U_{\text{ret}}(t) = u + c_{\text{ret}} t - \sum_{i=1}^{N(t)} X_{i, \text{ret}}$$

---

## Reinsurance Treaty Structures

### 1. Proportional (Quota-Share) Reinsurance

Under a proportional reinsurance treaty with retention proportion $\alpha \in [0.0, 1.0]$, the retained and ceded claim amounts are:

$$X_{\text{ret}} = \alpha X, \quad X_{\text{ced}} = (1 - \alpha) X$$

#### Retained Moments
- Mean retained severity:
  $$\mu_{1, \text{ret}}(\alpha) = \alpha \mu_1$$
- Retained variance:
  $$\sigma_{\text{ret}}^2(\alpha) = \alpha^2 \sigma^2$$
- Retained raw second moment:
  $$\mu_{2, \text{ret}}(\alpha) = \alpha^2 (\sigma^2 + \mu_1^2)$$

#### Retained Premium and Net Profit Condition
The retained premium rate is:

$$c_{\text{ret}}(\alpha) = (1 + \theta) \lambda \mu_1 - (1 + \xi) \lambda (1 - \alpha) \mu_1 = \lambda \mu_1 \left[ \theta - \xi(1 - \alpha) + \alpha \right]$$

The retained relative safety loading is:

$$\theta_{\text{ret}}(\alpha) = \frac{c_{\text{ret}}(\alpha)}{\lambda \mu_{1, \text{ret}}(\alpha)} - 1 = \frac{\theta - \xi(1 - \alpha)}{\alpha} \quad (\text{for } \alpha > 0)$$

To ensure solvency and prevent certain ruin ($\psi \equiv 1.0$), the net profit condition requires positive expected annual drift:

$$\mathbb{E}[U_{\text{ret}}(1) - u] = c_{\text{ret}}(\alpha) - \lambda \mu_{1, \text{ret}}(\alpha) = \lambda \mu_1 \left[ \theta - \xi(1 - \alpha) \right] > 0$$

which imposes a strict lower bound on retention:

$$\alpha > \alpha_{\min} = \max\left(0.0, 1.0 - \frac{\theta}{\xi}\right)$$

If $\xi \le \theta$, then $\alpha_{\min} = 0.0$ and any retention $\alpha \in (0, 1]$ is solvent. If $\xi > \theta$ (the reinsurer is more expensive than the direct market), ceding too much risk ($\alpha \le \alpha_{\min}$) creates negative drift, guaranteeing eventual ruin.

---

### 2. Excess-of-Loss (XL) Reinsurance

Under an Excess-of-Loss treaty with priority (attachment point) $d > 0$, the retained and ceded claim severities are:

$$X_{\text{ret}} = \min(X, d), \quad X_{\text{ced}} = (X - d)_+ = \max(0, X - d)$$

For exponential gross claim severities $X \sim \text{Exp}(\beta)$ with mean $\mu_1 = 1/\beta$:

#### Retained and Ceded Moments
- Expected ceded loss:
  $$\mathbb{E}[X_{\text{ced}}] = \int_d^\infty (x - d) \beta e^{-\beta x} dx = \mu_1 e^{-d / \mu_1}$$
- Retained expected severity:
  $$\mu_{1, \text{ret}}(d) = \mu_1 - \mathbb{E}[X_{\text{ced}}] = \mu_1 \left( 1 - e^{-d / \mu_1} \right)$$
- Retained raw second moment:
  $$\mu_{2, \text{ret}}(d) = \mathbb{E}[\min(X, d)^2] = 2 \mu_1^2 \left[ 1 - e^{-d / \mu_1} \left( 1 + \frac{d}{\mu_1} \right) \right]$$
- Retained variance:
  $$\sigma_{\text{ret}}^2(d) = \mu_{2, \text{ret}}(d) - [\mu_{1, \text{ret}}(d)]^2$$

#### Retained Premium Rate
$$c_{\text{ret}}(d) = (1 + \theta) \lambda \mu_1 - (1 + \xi) \lambda \mu_1 e^{-d / \mu_1}$$

---

## Lundberg Fundamental Equation & Adjustment Coefficient

The adjustment coefficient (Lundberg coefficient) $R > 0$ measures the exponential rate of decay of the ruin probability as initial capital $u \to \infty$. It is defined as the unique strictly positive root of Lundberg's fundamental equation:

$$\lambda \left( m_{X_{\text{ret}}}(R) - 1 \right) = c_{\text{ret}} R$$

### 1. Closed-Form Solution for Proportional Reinsurance

When individual claims follow an exponential distribution $X \sim \text{Exp}(\beta)$, the retained severity $X_{\text{ret}} = \alpha X$ follows an exponential distribution with rate $\beta / \alpha$ and mean $\alpha / \beta$.

The retained moment generating function is:

$$m_{\alpha X}(r) = \frac{\beta / \alpha}{\beta / \alpha - r} = \frac{\beta}{\beta - \alpha r} \quad (\text{for } r < \beta / \alpha)$$

Substituting into Lundberg's equation:

$$\lambda \left( \frac{\beta}{\beta - \alpha r} - 1 \right) = c_{\text{ret}} r \implies \lambda \frac{\alpha r}{\beta - \alpha r} = c_{\text{ret}} r$$

Canceling $r > 0$ yields the exact closed-form adjustment coefficient:

$$R(\alpha) = \frac{\beta}{\alpha} \cdot \frac{c_{\text{ret}} - \lambda \alpha \mu_1}{c_{\text{ret}}} = \frac{\beta}{\alpha} \cdot \frac{\theta_{\text{ret}}(\alpha)}{1 + \theta_{\text{ret}}(\alpha)}$$

Expressing in terms of market loadings and the retention proportion $\alpha$:

$$R(\alpha) = \frac{\beta [\theta - \xi(1 - \alpha)]}{\alpha [\alpha + \theta - \xi(1 - \alpha)]}$$

When $\alpha = 1$ (gross position, no reinsurance), $R(1) = \frac{\beta \theta}{1 + \theta}$, coinciding with classical Cramér-Lundberg theory. When $\alpha \le \alpha_{\min}$, net drift is non-positive and $R(\alpha) = 0.0$.

Substituting $R(\alpha)$ back into Lundberg's fundamental equation yields an exact residual of zero:

$$\lambda \left( m_{\alpha X}(R(\alpha)) - 1 \right) - c_{\text{ret}} R(\alpha) \equiv 0$$

### 2. Numerical Root-Finding for Excess-of-Loss Reinsurance

For Excess-of-Loss reinsurance with exponential claims, the retained claim severity $X_{\text{ret}} = \min(X, d)$ is bounded by $d$. Its moment generating function is:

$$m_{\min(X, d)}(r) = \int_0^d e^{rx} \beta e^{-\beta x} dx + e^{rd} \int_d^\infty \beta e^{-\beta x} dx = \frac{\beta - r e^{-(\beta - r)d}}{\beta - r}$$

At $r = \beta$, this expression possesses a removable singularity with limiting value:

$$\lim_{r \to \beta} m_{\min(X, d)}(r) = 1 + \beta d$$

The adjustment coefficient $R(d)$ is obtained by solving $g(r) = \lambda (m_{\min(X, d)}(r) - 1) - c_{\text{ret}} r = 0$ numerically via Brent's method (`scipy.optimize.brentq`).

---

## Optimal Reinsurance Retention

The optimal retention level maximizes the adjustment coefficient $R$, thereby minimizing the asymptotic ruin probability and maximizing long-term solvency:

$$R^* = \max_{\text{retention}} R$$

### Proportional Reinsurance Optimum
Differentiating $R(\alpha)$ with respect to $\alpha$ yields:

$$\frac{d R}{d \alpha} = \frac{\beta \left[ \xi(\alpha + \theta - \xi(1 - \alpha)) - (\theta - \xi(1 - \alpha))(1 + \xi) \right]}{[\alpha + \theta - \xi(1 - \alpha)]^2}$$

- **Case 1 ($\xi \le \theta$):** The reinsurer is cheaper or equal to direct premium loading. The derivative is non-positive for all $\alpha \in [0, 1]$. The optimal retention is full ceding down to the lowest permissible level.
- **Case 2 ($\xi > \theta$):** The reinsurer is more expensive. Setting $d R / d \alpha = 0$ yields the interior stationary point:
  $$\alpha^* = \frac{2 \xi - \theta}{\xi(1 + \xi)} \quad (\text{subject to } \alpha^* \in [\alpha_{\min}, 1.0])$$
  Our implementation optimizes this profile numerically via `scipy.optimize.minimize_scalar` within the robust bounded interval $[\alpha_{\min} + 10^{-4}(1 - \alpha_{\min}), 1.0]$, ensuring a strictly non-empty, well-posed search interval even when $\alpha_{\min} \to 1.0^-$.

---

## Ruin Probabilities & Lundberg Bounds

Given initial capital $u \ge 0$, the ultimate ruin probability is denoted by:

$$\psi(u) = \Pr\left( \inf_{t \ge 0} U_{\text{ret}}(t) < 0 \;\middle|\; U_{\text{ret}}(0) = u \right)$$

### 1. Lundberg Upper Bound
For any claim distribution with adjustment coefficient $R > 0$:

$$\psi(u) \le \exp(-R u)$$

### 2. Exact Cramér-Lundberg Ruin Probability (Exponential Claims)
When individual claim severities are exponentially distributed, the renewal equation admits an exact closed-form solution:

$$\psi_{\text{exact}}(u) = \begin{cases}
1.0 & \text{if } u = 0 \text{ or } \theta_{\text{ret}} \le 0 \\
\frac{1}{1 + \theta_{\text{ret}}} \exp(-R_{\text{ret}} u) & \text{if } u > 0 \text{ and } \theta_{\text{ret}} > 0
\end{cases}$$

Reinsurance increases $R_{\text{ret}} > R_{\text{gross}}$, creating exponential decay in ruin probability and providing significant risk reduction:

$$\text{Ruin Reduction Ratio} = 1 - \frac{\psi_{\text{ret}}(u)}{\psi_{\text{gross}}(u)}$$

---

## Solvency Capital Relief

For a regulatory ruin tolerance $\epsilon \in (0, 1)$ (e.g., Solvency II standard of $\epsilon = 0.005$ or 0.5% over 1 year):

$$u_{\epsilon} = \max\left(0.0, \frac{1}{R} \ln\left( \frac{1}{(1 + \theta)\epsilon} \right)\right)$$

Reinsurance reduces the required solvency capital from $u_{\epsilon}^{\text{gross}}$ to $u_{\epsilon}^{\text{re}}$. The capital relief generated by the reinsurance contract is:

$$\Delta u = u_{\epsilon}^{\text{gross}} - u_{\epsilon}^{\text{re}}$$

$$\text{Percentage Capital Relief} = \frac{\Delta u}{u_{\epsilon}^{\text{gross}}} \times 100\%$$

---

## Comparison: De Finetti Safety Index vs. Lundberg Adjustment Coefficient

| Feature | De Finetti Game / Diffusion Framework | Collective Risk Theory (Lundberg) |
|---|---|---|
| **Objective Criterion** | Maximizes Safety Index $S = \text{Drift} / \text{Variance}$ | Maximizes Adjustment Coefficient $R$ |
| **Penalty Structure** | Quadratic penalty on retained variance $\alpha^2 \sigma^2$ | Exponential penalty on all moments via MGF $m_X(r)$ |
| **Mathematical Tool** | Second-order diffusion approximation | Exact renewal equations / Martingales |
| **Tail Sensitivity** | Insensitive to heavy tails beyond 2nd moment | Sensitive to entire tail behavior |
| **Convergence** | Converges to Lundberg as claim jumps approach diffusion limit ($\mu_k \to 0$ for $k \ge 3$) | Exact benchmark for compound Poisson jump processes |

---

## Model Hypotheses and Domain Conditions

- **Poisson intensity:** $\lambda \in (0, \infty)$ strictly positive.
- **Claim moments:** Mean $\mu_1 \in (0, \infty)$ and variance $\sigma^2 \in (0, \infty)$ strictly positive.
- **Loadings:** Insurer safety loading $\theta \in (0, \infty)$, reinsurer loading $\xi \in [0, \infty)$.
- **Retention domain:** $\alpha \in [0.0, 1.0]$ for proportional, $d \in (0, \infty)$ for Excess-of-Loss.
- **Solvency drift:** Net drift $c_{\text{ret}} - \lambda \mu_{1, \text{ret}} > 0$. If net drift is non-positive, $R = 0.0$ and $\psi \equiv 1.0$.
- **Finite inputs:** All calculations strictly reject non-finite values (`NaN`, `Inf`, `-Inf`).

---

## Bibliographic References

1. **Kaas, R., Goovaerts, M., Dhaene, J., & Denuit, M. (2008).** *Modern Actuarial Risk Theory: Using R* (2nd ed.). Springer. Chapter 4: Ruin Theory (Sections 4.3 and 4.6).
2. **Bowers, N. L., Gerber, H. U., Hickman, J. C., Jones, D. A., & Nesbitt, C. J. (1997).** *Actuarial Mathematics* (2nd ed.). Society of Actuaries. Chapter 14: Collective Risk Models and Ruin Theory.
3. **Rolski, T., Schmidli, H., Schmidt, V., & Teugels, J. (1999).** *Stochastic Processes for Insurance and Finance*. John Wiley & Sons. Chapter 5: Ruin Probabilities.
4. **De Finetti, B. (1940).** *Il rischio nel giuoco e nelle assicurazioni*. IV Congresso dell'Unione Matematica Italiana.
