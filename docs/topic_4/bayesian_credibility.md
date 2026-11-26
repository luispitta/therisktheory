# Topic 4: Bayesian Credibility Theory

## 1. Theoretical Scope and Motivation
Bayesian credibility theory provides a coherent probabilistic framework for updating prior actuarial beliefs about an unobservable risk parameter $\lambda$ using observed portfolio claim experience $\mathbf{X} = (X_1, \dots, X_n)$. Under squared-error loss, the Bayes estimator of the risk premium coincides with the posterior expectation $E[\lambda \mid \mathbf{X}]$, yielding an exact linear credibility formula when prior and likelihood belong to conjugate families[cite: 1].

## 2. Mathematical Formulation and Proof of Conjugacy

### Model Assumptions
1. **Prior Distribution:** The unknown claim frequency rate $\lambda > 0$ follows a Gamma distribution with shape parameter $\alpha > 0$ and rate parameter $\beta > 0$:
   $$\pi(\lambda) = \frac{\beta^\alpha}{\Gamma(\alpha)} \lambda^{\alpha - 1} e^{-\beta \lambda}, \quad \lambda > 0$$
   with prior mean $\mu_0 = E[\lambda] = \frac{\alpha}{\beta}$ and prior variance $\text{Var}(\lambda) = \frac{\alpha}{\beta^2}$.

2. **Likelihood Function:** Conditioned on $\lambda$, the observed claim counts $X_1, \dots, X_n$ over known exposures $t_1, \dots, t_n > 0$ are independent Poisson random variables with $X_i \mid \lambda \sim \text{Poisson}(t_i \lambda)$:
   $$L(\lambda \mid \mathbf{X}) = \prod_{i=1}^n \frac{(t_i \lambda)^{X_i} e^{-t_i \lambda}}{X_i!} = \left(\prod_{i=1}^n \frac{t_i^{X_i}}{X_i!}\right) \lambda^{\sum_{i=1}^n X_i} e^{-\lambda \sum_{i=1}^n t_i}$$

### Derivation of the Posterior Distribution
By Bayes' theorem, the posterior density $\pi(\lambda \mid \mathbf{X})$ is proportional to the product of the prior density and the likelihood function:
$$\pi(\lambda \mid \mathbf{X}) \propto \pi(\lambda) \, L(\lambda \mid \mathbf{X})$$

Substituting both expressions and retaining only terms dependent on $\lambda$:
$$\pi(\lambda \mid \mathbf{X}) \propto \left(\lambda^{\alpha - 1} e^{-\beta \lambda}\right) \left(\lambda^{\sum_{i=1}^n X_i} e^{-\lambda \sum_{i=1}^n t_i}\right) = \lambda^{\left(\alpha + \sum_{i=1}^n X_i\right) - 1} e^{-\left(\beta + \sum_{i=1}^n t_i\right)\lambda}$$

Recognizing the kernel of a Gamma density, we obtain the exact conjugate posterior:
$$\lambda \mid \mathbf{X} \sim \text{Gamma}\left(\alpha_{\text{post}}, \beta_{\text{post}}\right)$$
where:
$$\alpha_{\text{post}} = \alpha + \sum_{i=1}^n X_i, \qquad \beta_{\text{post}} = \beta + \sum_{i=1}^n t_i$$

### Bayesian Credibility Premium and Credibility Factor $Z$
Under quadratic loss, the Bayesian credibility pure premium per unit of exposure is the posterior mean:
$$P_{\text{Bayes}} = E[\lambda \mid \mathbf{X}] = \frac{\alpha_{\text{post}}}{\beta_{\text{post}}} = \frac{\alpha + \sum_{i=1}^n X_i}{\beta + \sum_{i=1}^n t_i}$$

Let $T = \sum_{i=1}^n t_i$ be total exposure and $\bar{X} = \frac{\sum_{i=1}^n X_i}{T}$ be the observed empirical claim rate. Rewriting $P_{\text{Bayes}}$ as a convex combination:
$$P_{\text{Bayes}} = \frac{T}{\beta + T}\left(\frac{\sum_{i=1}^n X_i}{T}\right) + \frac{\beta}{\beta + T}\left(\frac{\alpha}{\beta}\right) = Z \bar{X} + (1 - Z)\mu_0$$
where the **Bayesian credibility weight** is:
$$Z = \frac{T}{\beta + T} \in (0, 1)$$
As $T \to \infty$, $Z \to 1$ (full weight on experience); as $T \to 0$, $Z \to 0$ (full weight on prior mean).

### Posterior Predictive Distribution and Mean
For a future exposure period $t_{\text{new}} > 0$, the conditional distribution of future claims is $X_{\text{new}} \mid \lambda \sim \text{Poisson}(t_{\text{new}} \lambda)$. Marginalizing over the posterior $\pi(\lambda \mid \mathbf{X})$ yields a Negative Binomial posterior predictive distribution with expectation (by the Law of Total Expectation):
$$E[X_{\text{new}} \mid \mathbf{X}] = E\big[E[X_{\text{new}} \mid \lambda, \mathbf{X}] \,\big|\, \mathbf{X}\big] = E[t_{\text{new}} \lambda \mid \mathbf{X}] = t_{\text{new}} \, P_{\text{Bayes}}$$
For a unit future exposure ($t_{\text{new}} = 1$), $E[X_{\text{new}} \mid \mathbf{X}] = P_{\text{Bayes}}$.

## 3. Bibliographic References
* **Bühlmann, H. (1970).** *Mathematical Methods in Risk Theory*. Springer-Verlag. (Chapter 4: Credibility Theory and Bayesian Estimation).
* **Klugman, S. A., Panjer, H. H., & Willmot, G. E. (2019).** *Loss Models: From Data to Decisions* (5th ed.). John Wiley & Sons. (Chapter 16: Bayesian Estimation and Conjugate Priors).
* **Bowers, N. L., Gerber, H. U., Hickman, J. C., Jones, D. A., & Nesbitt, C. J. (1997).** *Actuarial Mathematics* (2nd ed.). Society of Actuaries.
* **Straub, E. (1988).** *Non-Life Insurance Mathematics*. Springer-Verlag / Association of Swiss Actuaries.
