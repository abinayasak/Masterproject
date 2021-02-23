import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 400, 201)

w = np.load("fenics/solver/w.npy")
v = np.load("fenics/solver/v.npy")

plt.plot(t, v[:,0], label="v")
plt.plot(t, w[:,0], label="w")


plt.legend()
plt.title("FitzHugh-Nagumo Reparameterized model")
plt.show()
