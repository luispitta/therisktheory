# Topic 1: Risk Division, Retention, and Inflation Models

## Mathematical Concept Implemented
Principles of risk division, financial retention capacity, and the compound effects of inflation/deflation on an insurance risk portfolio (`retained_loss`, `retention_capacity`, `inflate_losses`, `deflate_losses`).

## Model Definitions & Hypotheses
- **Retained Loss**: Represents the portion of loss retained by the primary insurer under an excess-of-loss deductible or retention limit $M$:
  $$R(X) = \min(X, M)$$
- **Retention Capacity**: Evaluates the maximum financial retention based on available surplus $S$ and safety retention ratio $r \in [0, 1]$:
  $$C = S \times r$$
- **Inflation Adjustment**: Projects compound loss severity over $t$ discrete periods given a constant periodic inflation rate $i \ge -1$:
  $$L_{\text{inf}} = L_0 (1 + i)^t$$
- **Deflation / Present Value**: Discounts inflated loss amounts back to base period valuation:
  $$L_{\text{def}} = \frac{L_{\text{inf}}}{(1 + i)^t}$$

## Reversibility & Inversion
The inflation and deflation functions satisfy mathematical reversibility:
$$\text{deflate\_losses}(\text{inflate\_losses}(L, i, t), i, t) = L$$
for all valid parameters, provided $(1+i)^t \neq 0$.

## Domain & Boundary Conditions
- Losses $X \ge 0$, surplus $S \ge 0$, and retention limit $M \ge 0$.
- Retention ratio $0 \le r \le 1.0$.
- Inflation rate $i > -1.0$ (when periods $t > 0$ in `deflate_losses` to prevent division by zero).
- Periods $t \in \mathbb{Z}_{\ge 0}$.

## Bibliographic References
- Bühlmann, H. (1970). *Mathematical Methods in Risk Theory*. Springer-Verlag. (Chapter 4).
- Beard, R. E., Pentikäinen, T., & Pesonen, E. (1984). *Risk Theory*. 3rd ed., Chapman and Hall. (Chapter 2).
- Panjer, H. H., & Willmot, G. E. (1992). *Insurance Risk Models*. Society of Actuaries. (Chapter 2).
