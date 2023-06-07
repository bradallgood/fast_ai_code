
from matplotlib import pyplot as plt 
import numpy as np

# Define parabola
def f(x): 
    return x**2

# Define parabola derivative
def slope(x): 
    return 2*x

# Define x data range for parabola
x = np.linspace(-2,2,100)

# Choose point to plot tangent line
x1 = -1
y1 = f(x1)

# Define tangent line
# y = m*(x - x1) + y1
def line(x, x1, y1):
    return slope(x1)*(x - x1) + y1

# Define x data range for tangent line
xrange = np.linspace(x1-1, x1+1, 10)

# Plot the figure
plt.figure(figsize=(6,4))
plt.suptitle('x1=' + str(x1) + '   Slope=' + str(slope(x1)))
plt.grid()
plt.plot(x, f(x))
plt.scatter(x1, y1, color='C1', s=50)
plt.plot(xrange, line(xrange, x1, y1), 'C1--', linewidth = 2)

plt.show()