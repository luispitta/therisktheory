# Topic 5: Stability Parameters and Safety Loadings

Package: `therisktheory.collective_risk.stability`. Status: experimental.
References refer to Bühlmann, H. (1970), *Mathematical Methods in Risk Theory*, Springer; Beard, Pentikäinen and Pesonen (1984), *Risk Theory*, 3rd ed., Chapman & Hall; and Daykin, Pentikäinen and Pesonen (1994), *Practical Risk Theory for Actuaries*, Chapman & Hall.

## Mathematical Formulation
Let $U(t) = u + ct - S(t)$ with $c = (1 + \theta)\lambda \mu_1$ and annual result $G = c - S(1)$.

### 1. Stability Parameter
$$K = \frac{\mathbb{E}[G]}{\sigma(G)} = \theta \sqrt{\lambda} \frac{\mu_1}{\sqrt{\mu_2}}$$
(Bühlmann 1970, Section 6.4; Beard et al. 1984, Section 3.2).

### 2. Analytical Sensitivities
$$\frac{\partial K}{\partial \theta} = \frac{K}{\theta}, \quad \frac{\partial K}{\partial \lambda} = \frac{K}{2\lambda}, \quad \epsilon_\theta = 1.0, \quad \epsilon_\lambda = 0.5$$

### 3. Capital-Pricing Trade-off
Under the diffusion approximation with target ruin probability $\epsilon$:
$$\psi(u) \approx \exp\left(-\frac{2\theta\mu_1 u}{\mu_2}\right) \le \epsilon \implies \theta(u) = \frac{\mu_2 \ln(1 / \epsilon)}{2 \mu_1 u}$$
(Daykin et al. 1994, Section 8.3).
