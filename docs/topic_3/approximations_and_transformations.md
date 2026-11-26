# Topic 3: Gamma Approximations and Actuarial Transformations

## Theoretical Scope & Motivation
When evaluating aggregate loss distributions $S = \sum_{i=1}^N X_i$ in collective risk theory, calculating exact compound probabilities can be computationally expensive or intractable. Analytical approximations and variance-stabilizing transformations offer efficient moment-matched estimations for quantiles, Value-at-Risk (VaR), and premium re-weighting.

## Mathematical Formulations

### 1. Gamma Approximation (Moment Matching)
Approximates the aggregate loss $S$ with a continuous Gamma distribution $\text{Gamma}(\alpha, \theta)$ by matching the first two moments:
$$\alpha = \frac{\mathbb{E}[S]^2}{\text{Var}(S)}, \quad \theta = \frac{\text{Var}(S)}{\mathbb{E}[S]}$$

* **Domain & Restrictions**: $\mathbb{E}[S] > 0$ and $\text{Var}(S) > 0$.
* **Application**: Tail evaluation and quantile estimation for non-negative right-skewed aggregate losses.

### 2. Wilson–Hilferty Transformation
A cube-root transformation converting a Gamma or Chi-squared variable $X$ with $\nu$ degrees of freedom into an approximate standard normal variable $Z \sim \mathcal{N}(0, 1)$:
$$Z = \frac{\left(\frac{X}{\nu}\right)^{1/3} - \left(1 - \frac{2}{9\nu}\right)}{\sqrt{\frac{2}{9\nu}}}$$

* **Domain & Restrictions**: $\nu > 0$, $X \ge 0$.
* **Application**: Accelerates quantile computation without full numerical integration.

### 3. Haldane Transformation
A variance-stabilizing transformation applied to proportions or bounded claim frequencies $p \in (0, 1)$ over an exposure base $N$:
$$y = \ln\left(\frac{\tilde{p}}{1 - \tilde{p}}\right), \quad \text{where } \tilde{p} = \frac{p \cdot N + 0.5}{N + 1}$$

* **Domain & Restrictions**: $p \in [0, 1]$, $N > 0$.

### 4. Esscher Transformation (Exponential Tilting)
Re-weights a probability density or mass function $f(x)$ using an Esscher parameter $h \in \mathbb{R}$ to accentuate heavy-tail risks or perform change of measure:
$$f_h(x) = \frac{e^{hx} f(x)}{M_S(h)}$$
where $M_S(h) = \mathbb{E}[e^{hS}]$ is the moment-generating function evaluated at $h$.

* **Domain & Restrictions**: Requires $M_S(h) < \infty$ and $\sum f(x) = 1$.
* **Application**: Risk pricing, safety loading derivation, and extreme event probability adjustment.

## Bibliographic References
- Panjer, H. H. and Willmot, G. E. (1992). *Insurance Risk Models*. Society of Actuaries.
- Bühlmann, H. (1970). *Mathematical Methods in Risk Theory*. Springer-Verlag.
