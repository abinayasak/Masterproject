from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def dvdt(v, t):
    A = 0.04        # Constant
    v_rest = -85.0  # [mV]
    v_th = -65.0    # [mV]
    v_peak = 40.0   # [mV]
    return -A * A * (v - v_rest) * (v - v_th) * (v - v_peak)


def fitzhugh_nagumo(v, t):
    a = 0.13; b = 0.013              # Constant
    c1 = 0.26; c2 = 0.1; c3 = 1.0    # Constant
    i_app = 0.05                     # Constant

    dvdt = c1*v[0]*(v[0] - a)*(1 - v[0])
    dwdt = b*(v[0] - c3*v[1])

    dvdt += - c2*v[0]*v[1]   # Modified FHN
    #dvdt += - c2*v[1]       # Original FHN

    if t >= 50 and t <= 60:
        dvdt += i_app

    return dvdt, dwdt

def fitzhugh_nagumo_reparameterized(v, t):
    a = 0.13; b = 0.013              # Constant
    c1 = 0.26; c2 = 0.1; c3 = 1.0    # Constant
    i_app = 0.05                     # Constant

    v_rest = -85.            # [mV]
    v_peak = 40.             # [mV]
    v_amp = v_peak - v_rest  # [mV]
    v_th = v_rest + a*v_amp  # [mV]
    I_app = v_amp * i_app    # [mV]

    dVdt = c1*(v[0] - v_rest)*(v[0] - v_th)*(v_peak - v[0])/(v_amp**2) - (c2*(v[0] - v_rest)*v[1])/v_amp
    dWdt = b*(v[0] - v_rest - c3*v[1])


    if t >= 50 and t <= 60:
        dVdt += I_app

    return dVdt, dWdt


def monodomain_model(V, theta, u, v, u_n, dt):
    sigma_e = 0.62                             # [Sm^-1]
    sigma_i = 0.17                             # [Sm^-1]
    sigma = sigma_i*sigma_e/(sigma_e+sigma_i)  # [Sm^-1]
    chi = 140.                                 # [mm^-1]
    C_m = 0.01                                 # [mu*F*mm−2]
    M_i = (sigma)/(C_m*chi)

    if theta == 1:
        F = (
            u * v * dx
            + theta * (dt * dot(M_i * grad(u), grad(v)) * dx)
            - u_n * v * dx
        )
    else:
        F = (
            u * v * dx
            + theta * (dt * dot(M_i * grad(u), grad(v)) * dx)
            - u_n * v * dx
            + (1 - theta) * (dt * dot(M_i * grad(u_n), grad(v))) * dx
        )

    a, L = lhs(F), rhs(F)

    u = Function(V)  # u from step 2, inital value for step 3
    solve(a == L, u)

    return u


def step(V, T, N, dt, tn, Nx, Ny, degree, u0, w0, theta, derivative):
    u = TrialFunction(V)
    v = TestFunction(V)

    u0 = np.array(u0)
    w0 = np.array(w0)

    # Step one
    v_values = np.zeros(len(u0))
    w_values = np.zeros(len(u0))

    t = np.array([tn, tn + theta * dt])
    for i in range(len(u0)):
        v_values[i], w_values[i] = odeint(derivative, [u0[i], w0[i]], t)[-1]

    u_n = Function(V)
    u_n.vector()[:] = v_values

    # Step two
    #u = monodomain_model(V, theta, u, v, u_n, dt)

    # Step three
    if theta == 0.5:
        new_v_values= np.zeros(Nx + 1)
        new_w_values = np.zeros(Nx + 1)

        #u_n = u.vector()[:]
        u_n = u.vector()[:]
        t = np.array([tn + theta * dt, tn + dt])
        for i in range(Nx + 1):
            new_v_values[i], new_w_values[i] = odeint(derivative, [u_n[i], w_values[i]], t)[-1]

        u_new = Function(V)
        u_new.vector()[:] = new_v_values
        u = u_new


    return u_n, w_values



def run_solver(make_gif):

    theta = 1  # =0.5 Strang/CN and N must be large, =1 Godunov/BE
    degree = 1
    N = 400
    Nx = 600
    Ny = None
    T = 400.0                       # [ms]
    dt = T / N                      # [ms]
    t = np.linspace(0, T, N+1)      # [ms]

    mesh = IntervalMesh(Nx, 0, 20)  # [mm]
    V = FunctionSpace(mesh, "P", degree)
    #u0 = Expression('x[0] <= 2.0 ? 0 : -85', degree=0)
    #u0 = Expression('x[0] <= 2.0 ? -85 : -85', degree=0) # For plotting Reparameterized FHN
    u0 = Expression('x[0] <= 2.0 ? 0 : 0', degree=0)      # For plotting Original/Modified FHN
    u0 = interpolate(u0, V)

    u0 = u0.vector()[:]
    w0 = np.zeros(len(u0))

    x0 = Expression('x[0]', degree=0)
    x0 = interpolate(x0, V)
    np.save("x0", x0.vector()[:])


    derivative = fitzhugh_nagumo#_reparameterized

    v_list = []
    w_list = []
    t_list = []

    tn = 0
    count = 0
    skip_frames = 10

    v_list.append(u0[-1])
    w_list.append(w0[-1])
    t_list.append(tn)
    for i in range(N + 1):
        print("tn: %0.4f / %0.4f" % (tn, T))
        u, w = step(V, T, N, dt, tn, Nx, Ny, degree, u0, w0, theta, derivative)
        tn += dt
        u0 = u.vector()[:]
        w0 = w

        v_list.append(u0[-1])
        w_list.append(w[-1])
        t_list.append(tn)


        if make_gif:
            if i == count:
                # Create and save every skip_frames'th plots to file
                plt.clf()
                #plt.plot(np.load("x0.npy"), u.vector()[:], label="v")
                plt.plot(np.load("x0.npy"), u0, label="v")
                plt.plot(np.load("x0.npy"), w, label="w")
                #plt.plot(t, u0, label="v")
                #plt.plot(t, w, label="w")
                plt.axis([0, 20, -100, 100])
                #plt.xlabel("[mm]")
                #plt.ylabel("[mV]")
                plt.legend()
                plt.title("i=%d" % i)
                plt.savefig(f"plots/u{i:04d}.png")

                count += skip_frames

    np.save("v", v_list)
    np.save("w", w_list)
    np.save("t", t_list)


    if make_gif:

        import glob; import os
        from PIL import Image

        filepath_in = "plots/u*.png"
        filepath_out = "fitzhugh_nagumo_reparameterized_animation.gif"

        # Collecting the plots and putting them in a ordered list
        img, *imgs = [(Image.open(f)) for f in sorted(glob.glob(filepath_in))]

        # Create GIF
        img.save(fp=filepath_out,
            format="GIF",
            append_images=imgs,
            save_all=True,
            duration=200,
            loop=0,
        )

        # Delete all plots in file after creating the GIF
        [os.remove(f) for f in sorted(glob.glob(filepath_in))]


if __name__ == "__main__":
    run_solver(make_gif=False)
