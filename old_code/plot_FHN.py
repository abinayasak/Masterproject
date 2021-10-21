"""from fenics import *

mesh = Mesh('pre_torso.xml')

D = mesh.topology().dim()
subdomains = MeshFunction("size_t", mesh, D, mesh.domains())

vtkfile = File('mesh.pvd')
vtkfile << subdomains
"""

import numpy as np
import matplotlib.pyplot as plt


v = np.load('v.npy')
w = np.load('w.npy')
t = np.load('t.npy')


plt.plot(t, v, 'm', label='v')
plt.plot(t, w, 'k--', label='w')
plt.axis([0, 400, -0.5, 1])
plt.xlabel("[ms]")
#plt.ylabel("[mV]")

#plt.plot(x, bi_ue, label='Bidomain ue T-2dt')
#plt.plot(x, bi_coupled_ue, label='Bidomain Coupled ue T-2dt')

#plt.title("Reparameterized Fitzhugh-Nagumo model")
#plt.title("Original Fitzhugh-Nagumo model")
plt.title("Modified Fitzhugh-Nagumo model")
plt.grid()
plt.legend()
plt.show()
