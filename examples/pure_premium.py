import numpy as np

from therisktheory.solvency_pricing import pure_premium


frequency = np.array([0.5, 1.0, 2.0])
mean_severity = np.array([1000.0, 750.0, 500.0])

print(pure_premium(frequency, mean_severity))
