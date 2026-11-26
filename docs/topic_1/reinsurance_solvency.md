# Topic 1: Reinsurance and Solvency Evaluation

## Mathematical Concepts Implemented

Vectorized implementations of proportional and non-proportional reinsurance modalities, as well as cedent net solvency position evaluation under the `therisktheory.solvency_pricing` package:

1. `quota_share`: Proportional risk sharing with fixed retention rate $\alpha \in [0, 1]$.
2. `surplus_share`: Proportional treaty with retention line $L$ and treaty capacity limit $m \cdot L$.
3. `excess_of_loss`: Non-proportional per-risk / per-event reinsurance with priority $d$ and optional layer limit $l$.
4. `stop_loss`: Non-proportional aggregate portfolio cover with attachment deductible $D$ and optional capacity limit $L_{\text{cap}}$.
5. `reinsured_solvent_position`: Financial evaluation of insurer surplus, solvency indicator, and deficit after risk transfer.

---

## Model Definitions & Mathematical Formulations

### 1. Quota Share (Proportional)
For gross claim $X \ge 0$ and retention proportion $\alpha \in [0, 1]$:
$$X_{\text{ret}} = \alpha X, \quad X_{\text{ced}} = (1 - \alpha) X$$
Satisfies risk conservation: $X_{\text{ret}} + X_{\text{ced}} \equiv X$.

### 2. Surplus Share (Proportional Line Treaty)
For sum insured $S > 0$, monetary retention line $L > 0$, and maximum treaty lines $m \ge 0$:
$$\alpha = \min\left(1, \frac{L}{S}\right)$$
$$\beta = \min\left(1 - \alpha, \frac{m L}{S}\right)$$
$$X_{\text{ret}} = (1 - \beta) X, \quad X_{\text{ced}} = \beta X$$

### 3. Excess of Loss (Non-Proportional Per-Risk)
For claim $X \ge 0$, priority $d \ge 0$, and cover limit $l > 0$:
$$X_{\text{ced}} = \min\left(\max(0, X - d), l\right)$$
$$X_{\text{ret}} = X - X_{\text{ced}}$$

### 4. Stop-Loss (Non-Proportional Aggregate)
For aggregate portfolio loss $S \ge 0$, attachment point $D \ge 0$, and capacity limit $L_{\text{cap}} > 0$:
$$S_{\text{ced}} = \min\left(\max(0, S - D), L_{\text{cap}}\right)$$
$$S_{\text{ret}} = S - S_{\text{ced}}$$

### 5. Reinsured Solvent Position
Given initial reserves $u \ge 0$, retained premium $P_{\text{ret}} \ge 0$, and net retained claims $S_{\text{ret}} \ge 0$:
$$U_{\text{net}} = u + P_{\text{ret}} - S_{\text{ret}}$$
The insurer is solvent if $U_{\text{net}} \ge 0$. The insolvency deficit magnitude is $\max(0, -U_{\text{net}})$.

---

## Domain & Parameter Bounds
- Loss and claim amounts: $X \ge 0, S \ge 0$.
- Retention rate: $\alpha \in [0.0, 1.0]$.
- Priority / Deductibles: $d \ge 0, D \ge 0$.
- Cover limits / lines: $l > 0, L_{\text{cap}} > 0, L > 0, m \ge 0$.
- Capital and premiums: $u \ge 0, P_{\text{ret}} \ge 0$.

---

## Bibliographic References
- Kaas, R., Goovaerts, M., Dhaene, J., & Denuit, M. (2008). *Modern Actuarial Risk Theory: Using R* (2nd ed.). Springer, Section 1.3.
- Bowers, N. L., Gerber, H. U., Hickman, J. C., Jones, D. A., & Nesbitt, C. J. (1997). *Actuarial Mathematics* (2nd ed.). Society of Actuaries, Chapter 14.
- Klugman, S. A., Panjer, H. H., & Willmot, G. E. (2019). *Loss Models: From Data to Decisions* (5th ed.). Wiley.
