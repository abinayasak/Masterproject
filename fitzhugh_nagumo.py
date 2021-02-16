from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint



def solver(T, N, dt, tn, Nx, Ny, degree, u0, w0, theta):

    mesh = UnitIntervalMesh(Nx)
    V = FunctionSpace(mesh, "P", degree)

    u = TrialFunction(V)
    v = TestFunction(V)

    v_values = np.zeros(Nx + 1)
    w_values = np.zeros(Nx + 1)

    # Step one
    def fitzhugh_nagumo(v, t):
        a = 0.13; b = 0.013
        c1 = 0.26; c2 = 0.1; c3 = 1.0
        i_app = 0.5

        v_rest = -85
        v_peak = 40
        v_th = -68.75
        v_amp = v_peak - v_rest

        I_app = v_amp * i_app

        V = v_amp*v[0] + v_rest
        W = v_amp * v[1]

        dVdt = c1*(V - v_rest)*(V - v_th)*(v_peak - V)/(v_amp*v_amp) - c2*(V - v_rest)*W/v_amp
        dWdt = b*(V - v_rest - c3*v[1])

        #dVdt = c1*v[0]*(v[0] - a)*(1 - v[0]) - c2*v[1]
        #dWdt = b*(v[0] - c3*v[1])

        if t >= 50 and t <= 60 :
            dVdt += I_app

        return dVdt, dWdt


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

        for i in range(Nx + 1):
            v_values[i] = odeint(fitzhugh_nagumo, np.array([u0[i], w0[i]]), t)[-1][0]
            w_values[i] = odeint(fitzhugh_nagumo, np.array([u0[i], w0[i]]), t)[-1][-1]

    else:
        t = np.linspace(tn, tn + theta * dt, N)

        for i in range(Nx + 1):
            v_values[i] = odeint(fitzhugh_nagumo, np.array([u0[i], w0[i]]), t)[-1][0]
            w_values[i] = odeint(fitzhugh_nagumo, np.array([u0[i], w0[i]]), t)[-1][-1]


    u = v_values
    w = w_values

    """
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
    """

    return mesh, u, w


def run_solver(make_gif):

    theta = 1  # =0.5 Strang/CN and N must be large, =1 Godunov/BE
    N = 100
    Nx = 100
    Ny = 8
    T = 500.0  # [s]
    dt = T / N  # [s]
    degree = 1
    u0 = None
    w0 = np.linspace(0, 1, Nx+1)


    tn = 0
    count = 0
    skip_frames = 99

    t = np.arange(0, T+dt, dt)

    for i in range(N + 1):
        print("tn: %0.4f / %0.4f" % (tn, T))
        mesh, u, w = solver(T, N, dt, tn, Nx, Ny, degree, u0, w0, theta)
        tn += dt
        #u0 = u.vector()[:]
        u0 = u
        w0 = w

        if i == count:
            # Create and save every skip_frames'th plots to file
            plt.clf()
            plt.plot(t, u, label="v")
            plt.plot(t, w, label="w")
            #plt.axis([-0.5, T+0.5, -1, 1])
            plt.title("i=%d" % i)

            plt.savefig(f"plots/u{i:04d}.png")
            count += skip_frames


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
    run_solver(make_gif=True)
