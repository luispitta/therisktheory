# Topic 2: Applications of De Finetti Theory and Comparison with Classical Risk Theory

## Mathematical Concept Implemented
Applications of Bruno de Finetti's risk and game framework to elementary insurance, comparative analysis against classical risk theory, and evaluation of model assumptions, limitations, and structural criticisms (`compare_classical_de_finetti`, `elementary_insurance_case`).

## Model Definitions & Hypotheses
- **Elementary Insurance Case**: Formulates the insurance exchange under De Finetti's subjective probability and elementary game representations, contrasting utility-based approaches with collective risk surplus processes:
  $$\text{Terminal Position} = U_0 + P - (p \times S)$$
- **Classical vs. De Finetti Comparison**: Analyzes differences in ruin probabilities, reserve safety loadings, and capital requirements between classical Cramér-Lundberg formulations and De Finetti's discrete game models:
  $$\text{Deviation} = |\Pr(\text{Ruin}_{\text{Classical}}) - \Pr(\text{Ruin}_{\text{De Finetti}})|$$

## Domain & Boundary Conditions
- Probability weights $p \in [0, 1]$.
- Initial capital and reserve allocations $U_0 \ge 0$.
- Comparative parameters must share equivalent mean severity and premium income metrics to ensure valid benchmarking.

## Bibliographic References
- De Finetti, B. (1940). *Il rischio nel giuoco e nelle assicurazioni*. IV Congresso dell'Unione Matematica Italiana.
- Beard, R. E., Pentikäinen, T., & Pesonen, E. (1984). *Risk Theory*. Chapman and Hall.
- Kaas, R., Goovaerts, M., Dhaene, J., & Denuit, M. (2008). *Modern Actuarial Risk Theory: Using R*. Springer.
