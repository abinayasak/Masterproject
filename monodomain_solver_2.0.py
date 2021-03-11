from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def fitzhugh_nagumo_reparameterized(v, t):
    a = 0.13; b = 0.013            # Constant
    c1 = 0.26; c2 = 0.1; c3 = 1.0  # Constant
    i_app = 0.                     # Constant

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
    u = monodomain_model(V, theta, u, v, u_n, dt)


    # Step three
    if theta == 0.5:
        new_v_values= np.zeros(Nx + 1)
        new_w_values = np.zeros(Nx + 1)

        u_n = u.vector()[:]
        t = np.array([tn + theta * dt, tn + dt])
        for i in range(Nx + 1):
            new_v_values[i], new_w_values[i] = odeint(derivative, [u_n[i], w_values[i]], t)[-1]

        u_new = Function(V)
        u_new.vector()[:] = new_v_values
        u = u_new


    return u, w_values



def run_solver(make_gif, dimension):

    theta = 1  # =0.5 Strang/CN and N must be large, =1 Godunov/BE
    degree = 1
    N = 200
    Nx = 20
    Ny = 20
    T = 500.0                       # [ms]
    dt = T / N                      # [ms]
    t = np.linspace(0, T, N+1)      # [ms]

    if dimension == "1D":
        mesh = IntervalMesh(Nx, 0, 20)
    if dimension == "2D":
        mesh = RectangleMesh(Point(0, 0), Point(20, 20), Nx, Ny)
    V = FunctionSpace(mesh, "P", degree)
    u0 = Expression('x[0] <= 2.0 ? 0 : -85', degree=0)
    u_n = interpolate(u0, V)

    x0 = Expression('x[0]', degree=0)
    x0 = interpolate(x0, V)
    np.save("x0", x0.vector()[:])

    u0 = sorted(u_n.vector()[:])
    u0 = u0[::-1]
    w0 = np.zeros(len(u0))

    #set_initial_condition(V, mesh)
    derivative = fitzhugh_nagumo_reparameterized

    tn = 0
    count = 0
    skip_frames = 5
    for i in range(N + 1):
        print("tn: %0.4f / %0.4f" % (tn, T))
        u, w = step(V, T, N, dt, tn, Nx, Ny, degree, u0, w0, theta, derivative)
        tn += dt
        u0 = u.vector()[:]
        w0 = w

        if make_gif:
            if i == count:
                # Create and save every skip_frames'th plots to file
                plt.clf()
                if dimension == "1D":
                    plt.plot(x0.vector()[:], u.vector()[:][::-1], label="v")
                    plt.plot(x0.vector()[:], w[::-1], label="w")
                    plt.axis([0, 20, -100, 100])
                    plt.legend()
                    plt.xlabel("[mm]")
                    plt.ylabel("[mV]")

                if dimension == "2D":
                    c = plot(u, mode='color', vmin=-85, vmax=40)
                    plt.colorbar(c, orientation='vertical')
                    plt.xlabel("x [mm]")
                    plt.ylabel("y [mm]")

                plt.title("i=%d" % i)
                plt.savefig(f"plots/u{i:04d}.png")

                count += skip_frames


    if make_gif:

        import glob; import os
        from PIL import Image

        filepath_in = "plots/u*.png"
        filepath_out = "monodomain_%s.gif" % dimension

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
    run_solver(make_gif=True, dimension="2D")
