from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def fitzhugh_nagumo(v, t):
    a = 0.13; b = 0.013
    c1 = 0.26; c2 = 0.1; c3 = 1.0
    i_app = 0.05

    dvdt = c1*v[0]*(v[0] - a)*(1 - v[0]) - c2*v[1]
    dwdt = b*(v[0] - c3*v[1])

    if t >= 50 and t <= 60:
        dvdt += i_app

    return dvdt, dwdt



def set_initial_condition(Nx, V, mesh, tn, theta, dt):
    v_values = np.zeros(Nx + 1)
    w_values = np.zeros(Nx + 1)

    u0 = []
    element = V.element()
    for cell in cells(mesh):
        for i in range(element.tabulate_dof_coordinates(cell).size):
            # Discarding nodes that appears several time. Putting x coordinates into list u0
            if element.tabulate_dof_coordinates(cell)[i, 0] in u0:
                None
            else:
                u0.append(element.tabulate_dof_coordinates(cell)[i, 0])

    u0 = np.array(sorted(u0))  # Sorting the x coordinates as well as making u0 an array
    np.save("x0", u0)
    u0[u0 < 0.2] = 0.0  # if x < 0.2, set u0 = 0.
    u0[u0 >= 0.2] = -85.0  # if x >= 0.2, set u0 = -85.

    w0 = np.ones(len(u0))

    return u0, w0


def step(V, T, N, dt, tn, Nx, Ny, degree, u0, w0, theta):
    u = TrialFunction(V)
    v = TestFunction(V)

    v_values = np.zeros(Nx + 1)
    w_values = np.zeros(Nx + 1)

    t = np.array([tn, tn + theta * dt])

    for i in range(Nx + 1):
        v_values[i] = odeint(fitzhugh_nagumo, np.array([u0[i], w0[i]]), t)[-1][0]
        w_values[i] = odeint(fitzhugh_nagumo, np.array([u0[i], w0[i]]), t)[-1][-1]

    u_n = Function(V)
    u_n.vector()[:] = v_values
    w = w_values


    # Step two
    M_i = 1
    lmda = 0.004  # M_e = lmda * M_i
    gamma = float(dt * lmda / (1 + lmda))
    if theta == 1:
        F = (
            u * v * dx
            + theta * (gamma * dot(M_i * grad(u), grad(v)) * dx)
            - u_n * v * dx
        )
    else:
        F = (
            u * v * dx
            + theta * (gamma * dot(M_i * grad(u), grad(v)) * dx)
            - u_n * v * dx
            + (1 - theta) * (gamma * dot(M_i * grad(u_n), grad(v))) * dx
        )

    a, L = lhs(F), rhs(F)

    u = Function(V)  # u from step 2, inital value for step 3
    solve(a == L, u)


    # Step three
    if theta == 0.5:
        u_n = u.vector()[:]
        t = np.array([tn + theta * dt, tn + dt])

        for i in range(Nx + 1):
            v_values[i] = odeint(fitzhugh_nagumo, np.array([u_n[i], w0[i]]), t)[-1][0]
            w_values[i] = odeint(fitzhugh_nagumo, np.array([u_n[i], w0[i]]), t)[-1][-1]

        u_new = Function(V)
        u_new.vector()[:] = v_values
        u = u_new
        w = w_values



    return u, w


def run_step(make_gif):

    theta = 1  # =0.5 Strang/CN and N must be large, =1 Godunov/BE
    N = 100
    Nx = 100
    Ny = None
    T = 400  # [s]
    dt = T / N  # [s]
    degree = 1

    tn = 0
    count = 0
    skip_frames = 99

    t = np.linspace(0,T,N+1)


    mesh = UnitIntervalMesh(Nx)
    V = FunctionSpace(mesh, "P", degree)

    v_values = []
    w_values = []

    u0, w0 = set_initial_condition(Nx, V, mesh, tn, theta, dt)
    v_values.append(u0)
    w_values.append(w0)

    for i in range(N + 1):
        print("tn: %0.4f / %0.4f" % (tn, T))
        #print(u0)

        u, w = step(V, T, N, dt, tn, Nx, Ny, degree, u0, w0, theta)
        tn += dt

        u0 = u.vector()[:]
        w0 = w

        v_values.append(u.vector()[:])
        w_values.append(w)


        if i == count:
            # Create and save every skip_frames'th plots to file
            plt.clf()
            plt.plot(t, u.vector()[:], label="v")
            plt.plot(t, w, label="w")
            plt.axis([0, 400, -1, 1])
            plt.title("i=%d" % i)
            plt.legend()
            plt.savefig(f"plots/u{i:04d}.png")
            count += skip_frames

    np.save("v", v_values)
    np.save("w", w_values)

    if make_gif:

        import glob
        from PIL import Image
        import os

        filepath_in = "plots/u*.png"
        filepath_out = "animation.gif"

        # Collecting the plots and putting them in a list
        img, *imgs = [(Image.open(f)) for f in sorted(glob.glob(filepath_in))]

        # Create GIF
        img.save(fp=filepath_out,
            format="GIF",
            append_images=imgs[1:],
            save_all=True,
            duration=200,
            loop=0,
        )

        # Delete all plots in faile after creating the GIF
        [os.remove(f) for f in sorted(glob.glob(filepath_in))]


if __name__ == "__main__":
    #run_step(make_gif=True)
    run_step(make_gif=False)