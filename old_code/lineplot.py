import numpy as np
import matplotlib.pyplot as plt



v_bi = np.loadtxt('bi.txt')
v_mono = np.loadtxt('mono.txt')

x = np.linspace(0,20,len(v_bi))

plt.plot(x,v_bi,label='Bidomain')
plt.plot(x,v_mono,label='Monodomain')

plt.legend()
plt.show()
