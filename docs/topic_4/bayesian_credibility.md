# Topic 4: Bayesian Credibility Theory

## 1. Theoretical Scope and Motivation
Bayesian credibility theory provides a rigorous statistical framework for combining prior actuarial knowledge (prior distribution) with observed portfolio experience (likelihood function) to estimate the true risk premium for an individual policyholder or risk group.

## 2. Mathematical Formulation and Conjugacy
Let $\lambda$ be the risk parameter representing the underlying claim frequency for a given risk portfolio. 

* **Prior Distribution:** We assume a Gamma conjugate prior for $\lambda$:
  $$\pi(\lambda) = \frac{\beta^\alpha}{\Gamma(\alpha)} \lambda^{\alpha - 1} e^{-\beta \lambda}, \quad \alpha > 0, \beta > 0$$

* **Likelihood:** Given observed claim counts $X_i$ over exposure units $t_i$, the Poisson likelihood is:
  $$f(X_i \mid \lambda) = \frac{(t_i \lambda)^{X_i} e^{-t_i \lambda}}{X_i!}$$

* **Posterior Distribution:** By conjugacy, the posterior distribution of $\lambda$ given data $\mathbf{X}$ is also Gamma:
  $$\pi(\lambda \mid \mathbf{X}) = \text{Gamma}\left(\alpha + \sum X_i, \beta + \sum t_i\right)$$

* **Bayesian Credibility Premium:**
  $$P_{\text{Bayes}} = E[\lambda \mid \mathbf{X}] = \frac{\alpha + \sum X_i}{\beta + \sum t_i}$$

## 3. Bibliographic References
* Bühlmann, H. — *Mathematical Methods in Risk Theory*[cite: 1].
* Hogg, R.; Klugman, S. — *Loss Distributions*[cite: 1].
* Bowers, N.; Gerber, H.; Hickman, J.; Jones, D.; Nesbitt, C. — *Actuarial Mathematics*[cite: 1].
