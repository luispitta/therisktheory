import numpy as np

from therisktheory.solvency_pricing import expected_value_premium, pure_premium

frequency = np.array([0.5, 1.0, 2.0])
mean_severity = np.array([1000.0, 750.0, 500.0])

net = pure_premium(frequency, mean_severity)
print(net)  # [ 500.  750. 1000.]
print(expected_value_premium(net, loading=0.15))  # [ 575.   862.5 1150. ]
