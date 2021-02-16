from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

N = 10
Nx = 10

w0 = np.linspace(0, 1, Nx)

v0 = np.linspace(0, 1, Nx)
v0[v0 < 0.2] = 0.0
v0[v0 >= 0.2] = -85.0

def fitzhugh_nagumo(v, t):
    a = 0.13
    b = 0.013
    c1 = 0.26
    c2 = 0.1
    c3 = 1.0

    i_app = 0.05

    v_rest = -85
    v_peak = 40
    v_th = -68.75
    v_amp = v_peak - v_rest

    I_app = v_amp * i_app

    V = v_amp*v[0] + v_rest
    W = v_amp * v[1]

    dVdt = c1*(V - v_rest)*(V - v_th)*(v_peak - V)/(v_amp*v_amp) - c2*(V - v_rest)*W/(v_amp)
    dWdt = b*(V - v_rest - c3*v[1])

    #print(t)
    if t >= 50 and t <= 60 :
        #print("actication true")
        dVdt += I_app

    return dVdt, dWdt

t = np.linspace(1, 100, N)

initial_values = (v0,w0)
initial_values = np.vstack(initial_values)
#print(initial_values.shape)
#print(initial_values)

v_values, w_values = odeint(fitzhugh_nagumo, initial_values, t)

plt.plot(t, v_values, label="v")
plt.plot(t, w_values, label="w")
plt.legend()
plt.savefig("fitzhugh_nagumo.png")
