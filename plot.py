import numpy as np
import matplotlib.pyplot as plt

T = 1. * 1000 #[ms]
N = 100
dt = T/N

#print("x0:")
x0 = np.load("solver_array/x0.npy")
#print(x0)

#print("u0:")
u0 = np.load("solver_array/u0.npy")
#print(u0)

for i in range(0,N+1):
    plt.plot(x0, np.load("solver_array/u%d.npy"%i))

plt.title("Electrical Activity in Heart")
plt.xlabel("x")
plt.ylabel("u [mV]")
plt.show()
