import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Original data points
x_data = np.array([154, 355, 586, 1046, 1519, 2405])
y_data = np.array([100, 200, 300, 400, 500, 600])

# Calculate delta values
delta_x = np.diff(x_data)
delta_y = np.diff(y_data)
delta_rates = delta_y / delta_x
x_delta = (x_data[1:] + x_data[:-1]) / 2

# Calculate first delta
first_delta = y_data[0] / x_data[0]
x_delta = np.insert(x_delta, 0, x_data[0])
delta_rates = np.insert(delta_rates, 0, first_delta)

# Define pure exponential decay function


def decay_func(x, a, b):
    return a * np.exp(-b * x)


# Fit curve
popt, pcov = curve_fit(decay_func, x_delta, delta_rates, p0=[0.5, 0.0005])
a, b = popt

# Calculate integral function


def integral_func(x):
    return (a/b) * (1 - np.exp(-b * x))


# Get 99% point
x_99 = -np.log(0.01) / b  # 99% point
print(f"99% of relations requires {int(x_99)} claim pairs")

# Plot delta curve
plt.figure(figsize=(12, 8))
x_smooth = np.linspace(0, 10000, 1000)
y_smooth = decay_func(x_smooth, a, b)

# Add shaded area for 99%
plt.fill_between(x_smooth[x_smooth <= x_99], y_smooth[x_smooth <= x_99],
                 alpha=0.2, color='blue', label='99% Coverage Area')

plt.scatter(x_delta, delta_rates, color='red', s=100, label='Actual Delta')
plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='Fitted Delta Curve')

# Add vertical line at 99% point
plt.axvline(x=x_99, color='green', linestyle='--', alpha=0.5,
            label=f'99% Point ({int(x_99)} pairs)')

plt.xlabel('Claims Pairs')
plt.ylabel('Delta (New Relations/New Pair)')
plt.title('Delta Curve')
plt.legend()
plt.grid(True)
plt.xlim(0, 6000)
plt.ylim(0, max(delta_rates) * 1.1)

plt.tight_layout()
plt.show()

print(f"\nFitted function parameters:")
print(f"a = {a:.6f}")
print(f"b = {b:.6f}")
print(f"Theoretical maximum total relations: {integral_func(np.inf):.1f}")
