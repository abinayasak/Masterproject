import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 400, 201)

w = np.load("fenics/solver/w.npy")
v = np.load("fenics/solver/v.npy")

plt.plot(t, v, label="v")
plt.plot(t, w, label="w")
plt.legend()
plt.title("FitzHugh-Nagumo model")
plt.show()
