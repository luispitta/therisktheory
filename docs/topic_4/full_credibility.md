# Topic 4: Full Credibility Standards and Required Exposure

Package: `therisktheory.credibility.full_credibility`. Status: experimental.
References refer to Kaas, Goovaerts, Dhaene and Denuit (2008), *Modern Actuarial Risk Theory: Using R*, 2nd ed., Springer; and Klugman, Panjer and Willmot (2012), *Loss Models*, 4th ed., Wiley.

## Mathematical Formulation
Under classical limited fluctuation credibility, an observed experience is assigned full credibility ($Z = 1$) if the relative deviation between observed mean and expected mean does not exceed $k$ with probability $p = 1 - \alpha$:
$$\mathbb{P}(|X - \mu| \le k\mu) \ge p$$

### 1. Pure Frequency Standard (Poisson)
$$n_0 = \left(\frac{z_{1 - \alpha/2}}{k}\right)^2$$
For $k = 0.05$ and $p = 0.90 \implies n_0 \approx 1,082$ claims.
For $k = 0.05$ and $p = 0.95 \implies n_0 \approx 1,537$ claims.

### 2. Compound Aggregate Loss Standard
When individual claim severities $X$ are random:
$$n_S = n_0 \left(1 + \frac{\operatorname{Var}(X)}{(\mathbb{E}[X])^2}\right) = n_0 (1 + C_X^2)$$

### 3. Required Exposure Units
Given expected claim frequency rate $\lambda$ per exposure unit:
$$E_0 = \frac{n_S}{\lambda}$$
