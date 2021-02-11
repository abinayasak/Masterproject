from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
#import cbcbeat
#from IPython import display


def solver(T, N, dt, tn, Nx, Ny, degree, u0, theta):

    mesh = UnitIntervalMesh(Nx)
    V = FunctionSpace(mesh, "P", degree)

    u = TrialFunction(V)
    v = TestFunction(V)

    # Step one
    def dvdt(v, t):
        A = 0.04  # Constant
        v_rest = -85.0  # [mV]
        v_th = -65.0  # [mV]
        v_peak = 40.0  # [mV]
        return -A * A * (v - v_rest) * (v - v_th) * (v - v_peak)

    if tn == 0.0000:
        u0 = []
        element = V.element()
        for cell in cells(mesh):
            for i in range(element.tabulate_dof_coordinates(cell).size):
                # Discarding nodes that appears several time. Putting x coordinates into list u0
                if element.tabulate_dof_coordinates(cell)[i, 0] in u0:
                    None
                else:
                    u0.append(element.tabulate_dof_coordinates(cell)[i, 0])

        u0 = np.array(
            sorted(u0)
        )  # Sorting the x coordinates as well as making u0 an array
        np.save("x0", u0)
        u0[u0 < 0.2] = 0.0  # if x < 0.2, set u0 = 0.
        u0[u0 >= 0.2] = -85.0  # if x >= 0.2, set u0 = -85.

        t = np.linspace(tn, tn + theta * dt, N)
        u_ode = odeint(dvdt, u0, t)

    else:
        t = np.linspace(tn, tn + theta * dt, N)
        u_ode = odeint(dvdt, u0, t)

    u_n = Function(V)
    u_n.vector()[:] = u_ode[-1, :]  # u from step 1, inital value for step 2

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
        t = np.linspace(tn + theta * dt, tn + dt, N)
        u_ode = odeint(dvdt, u_n, t)

        u_new = Function(V)
        u_new.vector()[:] = u_ode[-1, :]
        u = u_new

    return mesh, u


def run_solver(make_gif):

    theta = 0.5  # =0.5 Strang/CN , =1 Godunov/BE
    N = 500
    Nx = 400
    Ny = 8
    T = 6.0  # [s]
    dt = T / N  # [s]
    degree = 1
    u0 = None

    tn = 0
    for i in range(N + 1):
        print("tn: %0.4f / %0.4f" % (tn, T))
        mesh, u = solver(T, N, dt, tn, Nx, Ny, degree, u0, theta)
        tn += dt
        u0 = u.vector()[:]

        # Create and save plots to file
        plt.clf()
        plt.plot(np.load("x0.npy"), u.vector())
        plt.axis([0, 1, -100, 50])
        plt.title("i=%d" % i)

        plt.savefig(f"plots/u{i:04d}.png")

    if make_gif:

        import glob
        from PIL import Image
        import os

        filepath_in = "plots/u*.png"
        filepath_out = "animation.gif"

        # Collecting the plots and putting them in a list
        img, *imgs = [(Image.open(f)) for f in sorted(glob.glob(filepath_in)) if f]

        # Make a new list with every 10th plot for the animation
        new_imgs = []
        for i in range(0, len(imgs), 10):
            new_imgs.append(imgs[i])

        # Create GIF
        img.save(fp=filepath_out,
            format="GIF",
            append_images=new_imgs,
            save_all=True,
            duration=200,
            loop=0,
        )

        # Delete all plots in faile after creating the GIF
        [os.remove(f) for f in sorted(glob.glob(filepath_in))]


if __name__ == "__main__":
    run_solver(make_gif=True)
